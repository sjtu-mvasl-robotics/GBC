import argparse
import getpass
import os
import stat
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm


DEFAULT_OUTPUT_DIR = "/cpfs/user/yaoyifei/datasets"
DEFAULT_CREDENTIALS_FILE = "~/.config/gbc/body_model_credentials.env"


@dataclass(frozen=True)
class BodyModelSpec:
    name: str
    url: str
    referer: str
    archive_name: str
    expected_files: tuple[str, ...]


MODEL_SPECS = {
    "smplh": BodyModelSpec(
        name="smplh",
        url="https://download.is.tue.mpg.de/download.php?domain=mano&resume=1&sfile=smplh.tar.xz",
        referer="https://mano.is.tue.mpg.de/download.php",
        archive_name="smplh.tar.xz",
        expected_files=(
            "smplh/neutral/model.npz",
            "smplh/male/model.npz",
            "smplh/female/model.npz",
        ),
    ),
    "dmpls": BodyModelSpec(
        name="dmpls",
        url="https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=dmpls.tar.xz",
        referer="https://smpl.is.tue.mpg.de/download.php",
        archive_name="dmpls.tar.xz",
        expected_files=(
            "dmpls/male/model.npz",
            "dmpls/female/model.npz",
            "dmpls/neutral/model.npz",
        ),
    ),
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


def resolve_credentials(args: argparse.Namespace, model: str) -> tuple[str, str]:
    file_values = read_env_file(args.credentials_file)
    upper_model = model.upper()

    username = (
        args.username
        or os.getenv(f"{upper_model}_USERNAME")
        or os.getenv("GBC_BODY_MODEL_USERNAME")
        or os.getenv("MPI_USERNAME")
        or file_values.get(f"{upper_model}_USERNAME")
        or file_values.get("GBC_BODY_MODEL_USERNAME")
        or file_values.get("MPI_USERNAME")
        or file_values.get("USERNAME")
    )
    password = (
        args.password
        or os.getenv(f"{upper_model}_PASSWORD")
        or os.getenv("GBC_BODY_MODEL_PASSWORD")
        or os.getenv("MPI_PASSWORD")
        or file_values.get(f"{upper_model}_PASSWORD")
        or file_values.get("GBC_BODY_MODEL_PASSWORD")
        or file_values.get("MPI_PASSWORD")
        or file_values.get("PASSWORD")
    )

    if not username:
        username = input(f"{model} username/email: ").strip()
    if not password:
        password = getpass.getpass(f"{model} password: ")

    if not username or not password:
        raise ValueError(f"Missing credentials for {model}")
    return username, password


def build_session(no_proxy: bool) -> requests.Session:
    session = requests.Session()
    if no_proxy:
        session.trust_env = False
    return session


def safe_extract_tar_xz(archive_path: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    with tarfile.open(archive_path, "r:xz") as tar:
        for member in tar.getmembers():
            target = (output_dir / member.name).resolve()
            if not str(target).startswith(str(output_dir) + os.sep):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        tar.extractall(output_dir)


def download_file(
    session: requests.Session,
    url: str,
    referer: str,
    username: str,
    password: str,
    output_path: Path,
    force: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Referer": referer}
    payload = {"username": username, "password": password, "commit": "Log in"}
    mode = "wb"
    existing_size = 0

    if output_path.exists() and not force:
        existing_size = output_path.stat().st_size
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

    response = session.post(url, data=payload, headers=headers, stream=True, timeout=120)
    if response.status_code == 416 and output_path.exists():
        print(f"{output_path.name} already appears complete.")
        return output_path
    if response.status_code == 404:
        raise RuntimeError(
            "Download returned 404. If this server requires a proxy in your network, "
            "source ~/.bashrc or export HTTP_PROXY/HTTPS_PROXY before retrying."
        )
    response.raise_for_status()

    if existing_size and response.status_code == 200:
        print(f"{output_path.name}: server ignored resume request; restarting download.")
        mode = "wb"
        existing_size = 0

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        raise RuntimeError(
            "Server returned HTML instead of an archive. Credentials may be invalid, "
            "or the account may not have accepted the required license on the model website."
        )

    total_size = int(response.headers.get("content-length", "0"))
    progress_total = existing_size + total_size if total_size else None
    with output_path.open(mode) as file_obj, tqdm(
        total=progress_total,
        initial=existing_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        desc=output_path.name.ljust(18),
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            written = file_obj.write(chunk)
            progress.update(written)

    return output_path


def find_existing_expected_files(output_dir: Path, spec: BodyModelSpec) -> list[Path]:
    return [output_dir / rel for rel in spec.expected_files if (output_dir / rel).exists()]


def summarize_expected_files(output_dir: Path, spec: BodyModelSpec) -> None:
    found = find_existing_expected_files(output_dir, spec)
    if found:
        print(f"{spec.name}: found model files:")
        for path in found:
            print(f"  - {path}")
    else:
        print(f"[WARNING] {spec.name}: no expected model.npz files found under {output_dir}")
        print("Checked:")
        for rel_path in spec.expected_files:
            print(f"  - {output_dir / rel_path}")


def download_model(
    spec: BodyModelSpec,
    output_dir: Path,
    archives_dir: Path,
    username: str,
    password: str,
    cleanup: bool,
    force: bool,
    no_proxy: bool,
) -> None:
    existing = find_existing_expected_files(output_dir, spec)
    if existing and not force:
        print(f"{spec.name}: expected files already exist, skipping download.")
        summarize_expected_files(output_dir, spec)
        return

    archive_path = archives_dir / spec.archive_name
    with build_session(no_proxy=no_proxy) as session:
        download_file(
            session=session,
            url=spec.url,
            referer=spec.referer,
            username=username,
            password=password,
            output_path=archive_path,
            force=force,
        )

    extract_dir = output_dir / spec.name
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} to {extract_dir}")
    safe_extract_tar_xz(archive_path, extract_dir)
    summarize_expected_files(output_dir, spec)

    if cleanup:
        archive_path.unlink(missing_ok=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SMPL+H and DMPL body model archives from MPI download servers.",
    )
    parser.add_argument("-u", "--username", help="MPI account email. Can also use env or credentials file.")
    parser.add_argument("-p", "--password", help="MPI account password. Prefer env or credentials file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to extract body models into. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--archives-dir",
        default=None,
        help="Directory for downloaded archives. Defaults to <output-dir>/_archives.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=sorted(MODEL_SPECS),
        help="Body model archives to download.",
    )
    parser.add_argument(
        "--credentials-file",
        default=DEFAULT_CREDENTIALS_FILE,
        help=f"Optional local credentials file. Default: {DEFAULT_CREDENTIALS_FILE}",
    )
    parser.add_argument("--cleanup", action="store_true", default=True, help="Delete archives after extraction.")
    parser.add_argument("--keep-archives", action="store_false", dest="cleanup", help="Keep downloaded archives.")
    parser.add_argument("--force", action="store_true", help="Force re-download and extraction.")
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Ignore HTTP_PROXY/HTTPS_PROXY from the environment.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    archives_dir = Path(args.archives_dir).expanduser().resolve() if args.archives_dir else output_dir / "_archives"
    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        spec = MODEL_SPECS[model]
        if find_existing_expected_files(output_dir, spec) and not args.force:
            print("-" * 60)
            print(f"{spec.name}: expected files already exist, skipping download.")
            summarize_expected_files(output_dir, spec)
            continue
        username, password = resolve_credentials(args, model)
        print("-" * 60)
        print(f"Processing {spec.name}")
        download_model(
            spec=spec,
            output_dir=output_dir,
            archives_dir=archives_dir,
            username=username,
            password=password,
            cleanup=args.cleanup,
            force=args.force,
            no_proxy=args.no_proxy,
        )

    print("=" * 60)
    print(f"Body models are available under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
