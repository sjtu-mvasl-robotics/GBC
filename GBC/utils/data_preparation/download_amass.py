import argparse
import getpass
import os
import re
import stat
import sys
import tarfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE_URL = "https://amass.is.tue.mpg.de/"
LOGIN_URL = urljoin(BASE_URL, "login.php")
DOWNLOAD_PAGE_URL = urljoin(BASE_URL, "download.php")
DEFAULT_OUTPUT_DIR = "/cpfs/user/yaoyifei/datasets/AMASS_FULL"
DEFAULT_CREDENTIALS_FILE = "~/.config/gbc/amass_credentials.env"
COMPLETION_MARKER_DIR = ".amass_download_complete"
LEGACY_EXTRACTED_DIRS = {
    "DFaust.tar.bz2": ("DFaust_67",),
}


def read_env_file(path: str) -> dict[str, str]:
    expanded = Path(path).expanduser()
    if not expanded.exists():
        return {}

    mode = stat.S_IMODE(expanded.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(
            f"[WARNING] Credentials file {expanded} is readable by group/others. "
            "Run: chmod 600 <file>",
            file=sys.stderr,
        )

    values: dict[str, str] = {}
    for raw_line in expanded.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def resolve_credentials(args: argparse.Namespace) -> tuple[str, str]:
    file_values = read_env_file(args.credentials_file)
    username = (
        args.username
        or os.getenv("AMASS_USERNAME")
        or os.getenv("GBC_AMASS_USERNAME")
        or file_values.get("AMASS_USERNAME")
        or file_values.get("GBC_AMASS_USERNAME")
        or file_values.get("USERNAME")
    )
    password = (
        args.password
        or os.getenv("AMASS_PASSWORD")
        or os.getenv("GBC_AMASS_PASSWORD")
        or file_values.get("AMASS_PASSWORD")
        or file_values.get("GBC_AMASS_PASSWORD")
        or file_values.get("PASSWORD")
    )

    if not username:
        username = input("AMASS username/email: ").strip()
    if not password:
        password = getpass.getpass("AMASS password: ")

    if not username or not password:
        raise ValueError("Missing AMASS credentials")
    return username, password


def build_session(no_proxy: bool) -> requests.Session:
    session = requests.Session()
    if no_proxy:
        session.trust_env = False
    return session


def login(session: requests.Session, username: str, password: str, quiet: bool = False) -> bool:
    if not quiet:
        print(f"Attempting to log in as {username}...")
    login_payload = {"username": username, "password": password, "commit": "Log in"}
    try:
        response = session.post(LOGIN_URL, data=login_payload, timeout=60)
        response.raise_for_status()
        verify_response = session.get(DOWNLOAD_PAGE_URL, timeout=60)
        verify_response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"\n[ERROR] An error occurred during login: {exc}")
        return False

    if "Sign in" in verify_response.text:
        print("\n[ERROR] Login failed. Please check your username and password.")
        return False

    if not quiet:
        print("Login successful.")
    return True


def get_download_urls(
    session: requests.Session,
    smplh_only: bool,
    variant: str,
    specialize: str | None = None,
) -> list[str]:
    print("Finding available AMASS dataset files...")
    try:
        response = session.get(DOWNLOAD_PAGE_URL, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"\n[ERROR] Could not fetch download page: {exc}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    urls: list[str] = []
    buttons = soup.find_all("button", onclick=re.compile(r"openModalLicense"))
    for button in buttons:
        onclick_attr = button.get("onclick", "")
        match = re.search(r"openModalLicense\('([^']+)'", onclick_attr)
        if not match:
            continue

        url = urljoin(BASE_URL, match.group(1).replace("&amp;", "&"))
        if "/mosh_results/" not in url:
            continue
        if smplh_only and "/smplh/" not in url:
            continue
        if smplh_only and variant != "all" and f"/smplh/{variant}/" not in url:
            continue
        if specialize and specialize not in url:
            continue
        urls.append(url)

    urls = sorted(set(urls))
    if not urls:
        print("[ERROR] Could not find any download links. The page structure may have changed.")
        return []

    filter_msg = "smplh" if smplh_only else "all mosh_results"
    if smplh_only:
        filter_msg += f"/{variant}"
    if specialize:
        filter_msg += f", specialize={specialize}"
    print(f"Found {len(urls)} unique files to download ({filter_msg}).")
    return urls


def filename_from_url(url: str) -> str:
    parsed_url = urlparse(url)
    query = parse_qs(parsed_url.query)
    sfile = query.get("sfile", [None])[0]
    if not sfile:
        raise ValueError(f"Could not parse filename from URL: {url}")
    return os.path.basename(unquote(sfile))


def archive_dataset_name(archive_name: str) -> str:
    for suffix in (".tar.bz2", ".zip"):
        if archive_name.endswith(suffix):
            return archive_name[: -len(suffix)]
    return Path(archive_name).stem


def completion_marker(output_dir: Path, archive_name: str) -> Path:
    return output_dir / COMPLETION_MARKER_DIR / f"{archive_name}.done"


def archive_looks_extracted(output_dir: Path, archive_name: str, archive_path: Path | None = None) -> bool:
    if completion_marker(output_dir, archive_name).exists():
        return True

    if archive_path is not None and archive_path.exists():
        return False

    candidates = [archive_dataset_name(archive_name)]
    candidates.extend(LEGACY_EXTRACTED_DIRS.get(archive_name, ()))
    for candidate in candidates:
        candidate_dir = output_dir / candidate
        if candidate_dir.is_dir() and any(candidate_dir.rglob("*.npz")):
            return True
    return False


def mark_archive_complete(output_dir: Path, archive_name: str) -> None:
    marker = completion_marker(output_dir, archive_name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n")


def get_remote_size(session: requests.Session, url: str, headers: dict[str, str]) -> int:
    try:
        response = session.head(url, headers=headers, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return 0
    return int(response.headers.get("content-length", "0"))


def download_file(
    session: requests.Session,
    url: str,
    archives_dir: Path,
    force: bool = False,
    progress_position: int = 0,
    leave_progress: bool = True,
) -> Path | None:
    try:
        local_filename = filename_from_url(url)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}")
        return None

    archives_dir.mkdir(parents=True, exist_ok=True)
    local_filepath = archives_dir / local_filename
    headers = {"Referer": DOWNLOAD_PAGE_URL}
    total_size = get_remote_size(session, url, headers)
    local_file_size = 0
    file_mode = "wb"

    if local_filepath.exists() and not force:
        local_file_size = local_filepath.stat().st_size
        if total_size and local_file_size >= total_size:
            print(f"{local_filename} already complete. Skipping download.")
            return local_filepath
        if local_file_size > 0:
            print(f"Resuming {local_filename} from {local_file_size / 1024**2:.2f} MiB...")
            headers["Range"] = f"bytes={local_file_size}-"
            file_mode = "ab"
    else:
        print(f"\nStarting download for {local_filename}...")

    try:
        with session.get(url, headers=headers, stream=True, timeout=120) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("content-type", ""):
                print(f"\n[ERROR] Server returned an HTML page for {local_filename}. Skipping.")
                return None

            if local_file_size and response.status_code == 200:
                print(f"{local_filename}: server ignored resume request; restarting download.")
                local_file_size = 0
                file_mode = "wb"

            if not total_size:
                total_size = int(response.headers.get("content-length", "0"))
            progress_total = local_file_size + int(response.headers.get("content-length", "0")) if total_size else None

            with local_filepath.open(file_mode) as file_obj, tqdm(
                total=progress_total,
                initial=local_file_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                desc=local_filename.ljust(25),
                position=progress_position,
                leave=leave_progress,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    written = file_obj.write(chunk)
                    progress_bar.update(written)
    except requests.exceptions.RequestException as exc:
        print(f"\n[ERROR] Could not download {local_filename}: {exc}")
        return None

    if total_size and local_filepath.stat().st_size < total_size:
        print(f"\n[WARNING] Download for {local_filename} may be incomplete.")
        return None

    return local_filepath


def safe_extract_tar_bz2(filepath: Path, extract_dir: Path) -> bool:
    print(f"Extracting {filepath.name}...")
    extract_dir = extract_dir.resolve()
    try:
        with tarfile.open(filepath, "r:bz2") as tar:
            for member in tar.getmembers():
                target = (extract_dir / member.name).resolve()
                if not str(target).startswith(str(extract_dir) + os.sep):
                    raise RuntimeError(f"Unsafe archive member path: {member.name}")
            tar.extractall(path=extract_dir)
        print("Extraction complete.")
        return True
    except (tarfile.TarError, RuntimeError) as exc:
        print(f"[ERROR] Failed to extract {filepath.name}: {exc}")
        return False


def safe_extract_zip(filepath: Path, extract_dir: Path) -> bool:
    print(f"Extracting {filepath.name}...")
    extract_dir = extract_dir.resolve()
    try:
        with zipfile.ZipFile(filepath) as archive:
            for member in archive.infolist():
                target = (extract_dir / member.filename).resolve()
                if not str(target).startswith(str(extract_dir) + os.sep):
                    raise RuntimeError(f"Unsafe archive member path: {member.filename}")
            archive.extractall(path=extract_dir)
        print("Extraction complete.")
        return True
    except (zipfile.BadZipFile, RuntimeError) as exc:
        print(f"[ERROR] Failed to extract {filepath.name}: {exc}")
        return False


def extract_archive(filepath: Path, extract_dir: Path) -> bool:
    suffixes = "".join(filepath.suffixes)
    if suffixes.endswith(".tar.bz2"):
        return safe_extract_tar_bz2(filepath, extract_dir)
    if filepath.suffix == ".zip":
        return safe_extract_zip(filepath, extract_dir)

    print(f"[ERROR] Unsupported archive type: {filepath.name}")
    return False


def download_extract_one(
    index: int,
    total: int,
    url: str,
    username: str,
    password: str,
    output_dir: Path,
    archives_dir: Path,
    force: bool,
    cleanup: bool,
    no_proxy: bool,
    extract_lock: threading.Lock,
    progress_position: int = 0,
    leave_progress: bool = False,
    download_retries: int = 1,
    retry_sleep: float = 30.0,
) -> tuple[str, bool]:
    local_filename = filename_from_url(url)
    local_archive = archives_dir / local_filename
    if archive_looks_extracted(output_dir, local_filename, local_archive) and not force:
        print(f"{local_filename}: extracted dataset already exists, skipping.")
        mark_archive_complete(output_dir, local_filename)
        if cleanup:
            local_archive.unlink(missing_ok=True)
        return local_filename, True

    print("-" * 60)
    print(f"Processing file {index}/{total}: {local_filename}")

    downloaded_path = None
    attempts = max(1, download_retries)
    for attempt in range(1, attempts + 1):
        with build_session(no_proxy=no_proxy) as session:
            if not login(session, username, password, quiet=True):
                print(f"[ERROR] Worker login failed for {local_filename}")
            else:
                downloaded_path = download_file(
                    session,
                    url,
                    archives_dir,
                    force,
                    progress_position=progress_position,
                    leave_progress=leave_progress,
                )

        if downloaded_path:
            break
        if attempt < attempts:
            delay = max(0.0, retry_sleep)
            print(f"[RETRY] {local_filename}: retrying download {attempt + 1}/{attempts} in {delay:g}s.")
            if delay:
                time.sleep(delay)

    if not downloaded_path:
        return local_filename, False

    with extract_lock:
        if extract_archive(downloaded_path, output_dir):
            mark_archive_complete(output_dir, local_filename)
            if cleanup:
                print(f"Cleaning up {downloaded_path.name}...")
                downloaded_path.unlink(missing_ok=True)
            return local_filename, True

    print(f"Skipping cleanup for {downloaded_path.name} due to extraction error.")
    return local_filename, False


def download_extract_many(
    urls: list[str],
    username: str,
    password: str,
    output_dir: Path,
    archives_dir: Path,
    force: bool,
    cleanup: bool,
    no_proxy: bool,
    num_workers: int,
    download_retries: int,
    retry_sleep: float,
) -> int:
    extract_lock = threading.Lock()
    failures = 0
    total = len(urls)

    if num_workers <= 1:
        for index, url in enumerate(urls, 1):
            _, ok = download_extract_one(
                index=index,
                total=total,
                url=url,
                username=username,
                password=password,
                output_dir=output_dir,
                archives_dir=archives_dir,
                force=force,
                cleanup=cleanup,
                no_proxy=no_proxy,
                extract_lock=extract_lock,
                progress_position=0,
                leave_progress=True,
                download_retries=download_retries,
                retry_sleep=retry_sleep,
            )
            failures += 0 if ok else 1
        return failures

    print(f"Downloading with {num_workers} parallel workers.")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                download_extract_one,
                index,
                total,
                url,
                username,
                password,
                output_dir,
                archives_dir,
                force,
                cleanup,
                no_proxy,
                extract_lock,
                (index - 1) % num_workers,
                False,
                download_retries,
                retry_sleep,
            ): url
            for index, url in enumerate(urls, 1)
        }
        for future in as_completed(futures):
            try:
                local_filename, ok = future.result()
            except Exception as exc:
                local_filename = filename_from_url(futures[future])
                ok = False
                print(f"[ERROR] {local_filename} failed unexpectedly: {exc}")
            if ok:
                print(f"[DONE] {local_filename}")
            else:
                print(f"[FAILED] {local_filename}")
                failures += 1

    return failures


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downloader and extractor for AMASS mosh_results archives.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-u", "--username", help="AMASS account email. Can also use env or credentials file.")
    parser.add_argument("-p", "--password", help="AMASS account password. Prefer env or credentials file.")
    parser.add_argument(
        "-o",
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to extract AMASS files to. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--archives-dir",
        default=None,
        help="Directory for partial/downloaded archives. Defaults to <output_dir>.",
    )
    parser.add_argument(
        "--smplh_only",
        action="store_true",
        default=True,
        help="Only download files from the smplh category. Default: true.",
    )
    parser.add_argument(
        "--all-mosh-results",
        action="store_false",
        dest="smplh_only",
        help="Download every mosh_results archive instead of only smplh archives.",
    )
    parser.add_argument(
        "--variant",
        choices=("gender_specific", "neutral", "all"),
        default="gender_specific",
        help="SMPLH archive variant to download when --smplh_only is active. Default: gender_specific.",
    )
    parser.add_argument(
        "--gender",
        choices=("neutral", "all"),
        dest="variant",
        help="Deprecated alias for --variant kept for older local commands.",
    )
    parser.add_argument("--specialize", help="Download only dataset URLs containing this string, e.g. 'CMU'.")
    parser.add_argument(
        "--credentials-file",
        default=DEFAULT_CREDENTIALS_FILE,
        help=f"Optional local credentials file. Default: {DEFAULT_CREDENTIALS_FILE}",
    )
    parser.add_argument("--cleanup", action="store_true", default=True, help="Delete archives after extraction.")
    parser.add_argument("--keep-archives", action="store_false", dest="cleanup", help="Keep archives after extraction.")
    parser.add_argument("--force", action="store_true", help="Force re-download of existing archives.")
    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=1,
        help="Number of archives to download in parallel. Default: 1.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=10,
        help="Per-archive resume retries after transient network failures. Default: 10.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=30.0,
        help="Seconds to wait between per-archive retries. Default: 30.",
    )
    parser.add_argument("--list-only", action="store_true", help="Only list matching archive URLs; do not download.")
    parser.add_argument("--no-proxy", action="store_true", help="Ignore HTTP_PROXY/HTTPS_PROXY from the environment.")
    args = parser.parse_args(argv)
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if args.download_retries < 1:
        parser.error("--download-retries must be >= 1")
    if args.retry_sleep < 0:
        parser.error("--retry-sleep must be >= 0")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    archives_dir = Path(args.archives_dir).expanduser().resolve() if args.archives_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir.mkdir(parents=True, exist_ok=True)
    username, password = resolve_credentials(args)

    with build_session(no_proxy=args.no_proxy) as session:
        if not login(session, username, password):
            return 1

        download_urls = get_download_urls(session, args.smplh_only, args.variant, args.specialize)
        if not download_urls:
            return 1

        if args.list_only:
            for url in download_urls:
                print(f"- {filename_from_url(url)}\t{url}")
            return 0

    failures = download_extract_many(
        urls=download_urls,
        username=username,
        password=password,
        output_dir=output_dir,
        archives_dir=archives_dir,
        force=args.force,
        cleanup=args.cleanup,
        no_proxy=args.no_proxy,
        num_workers=args.num_workers,
        download_retries=args.download_retries,
        retry_sleep=args.retry_sleep,
    )

    print("\n" + "=" * 60)
    print("All tasks completed.")
    print(f"AMASS dataset is ready in: {output_dir}")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
