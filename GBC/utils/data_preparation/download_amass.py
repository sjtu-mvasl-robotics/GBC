import requests
import sys
import os
import tarfile
from tqdm import tqdm
import argparse
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

# --- Configuration ---
BASE_URL = 'https://amass.is.tue.mpg.de/'
LOGIN_URL = f'{BASE_URL}login.php'
DOWNLOAD_PAGE_URL = f'{BASE_URL}download.php'

def login(session, username, password):
    """Logs into the AMASS website and returns the session object."""
    print(f"Attempting to log in as {username}...")
    login_payload = {'username': username, 'password': password, 'commit': 'Log in'}
    try:
        r = session.post(LOGIN_URL, data=login_payload, timeout=30)
        r.raise_for_status()
        verify_r = session.get(DOWNLOAD_PAGE_URL, timeout=30)
        if 'Sign in' in verify_r.text:
            print("\n[ERROR] Login failed. Please check your username and password.")
            return None
        print("Login successful.")
        return session
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] An error occurred during login: {e}")
        return None

def get_download_urls(session, smplh_only, specialize=None):
    """Scrapes the downloads page to find all available download URLs."""
    print("Finding all available dataset files...")
    try:
        r = session.get(DOWNLOAD_PAGE_URL, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
        urls = []
        buttons = soup.find_all('button', onclick=re.compile(r'openModalLicense'))
        for button in buttons:
            onclick_attr = button['onclick']
            match = re.search(r"openModalLicense\('([^']+)'", onclick_attr)
            if match:
                url = match.group(1).replace('&amp;', '&')
                if not '/mosh_results/' in url: # remove visual results
                    continue
                if smplh_only and '/smplh/' not in url:
                    continue
                if specialize and specialize not in url:
                    continue
                urls.append(url)
        if not urls:
            print("[ERROR] Could not find any download links. The page structure may have changed.")
            return []
        urls = sorted(list(set(urls)))
        filter_msg = " (filtered for 'smplh' only)" if smplh_only else ""
        print(f"Found {len(urls)} unique files to download{filter_msg}.")
        return urls
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Could not fetch download page: {e}")
        return []

def download_file(session, url, output_dir, force=False):
    """Downloads a single file with progress bar, resume logic, and the crucial Referer header."""
    try:
        # Extract filename from the 'sfile' parameter in the URL
        parsed_url = urlparse(url)
        sfile = parsed_url.query.split('sfile=')[1].split('&')[0]
        local_filename = os.path.basename(sfile)
    except IndexError:
        print(f"\n[ERROR] Could not parse filename from URL: {url}")
        return None
        
    local_filepath = os.path.join(output_dir, local_filename)
    
    # *** THIS IS THE CRITICAL FIX ***
    # The download server requires the Referer to prove the request is legitimate.
    headers = {'Referer': DOWNLOAD_PAGE_URL}
    
    local_file_size = 0
    file_mode = 'wb'

    try:
        # Check total file size with a HEAD request first
        with session.head(url, headers=headers, timeout=30) as r_head:
            r_head.raise_for_status()
            total_size = int(r_head.headers.get('content-length', 0))

        # Implement Resume Logic
        if os.path.exists(local_filepath) and not force:
            local_file_size = os.path.getsize(local_filepath)
            if total_size != 0 and local_file_size >= total_size:
                print(f"'{local_filename}' already complete. Skipping.")
                return local_filepath
            
            print(f"Resuming download for '{local_filename}' from {local_file_size / 1024**2:.2f} MB...")
            headers['Range'] = f'bytes={local_file_size}-'
            file_mode = 'ab'
        else:
            print(f"\nStarting download for {local_filename}...")

        # Perform the actual download
        with session.get(url, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()

            if 'text/html' in r.headers.get('Content-Type', ''):
                print(f"\n[ERROR] Authentication failed. Server returned an HTML page for {local_filename}. Skipping.")
                return None
            
            with open(local_filepath, file_mode) as f, tqdm(
                total=total_size, initial=local_file_size,
                unit='iB', unit_scale=True, unit_divisor=1024,
                desc=local_filename.ljust(25) # Pad filename for alignment
            ) as progress_bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        progress_bar.update(size)
        
        if total_size != 0 and os.path.getsize(local_filepath) < total_size:
             print(f"\n[WARNING] Download for {local_filename} may be incomplete.")
             return None
             
        return local_filepath
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Could not download {local_filename}: {e}")
        return None

def extract_archive(filepath, extract_dir):
    """Extracts a .tar.bz2 file."""
    print(f"Extracting {os.path.basename(filepath)}...")
    try:
        with tarfile.open(filepath, 'r:bz2') as tar:
            tar.extractall(path=extract_dir)
        print("Extraction complete.")
        return True
    except tarfile.TarError as e:
        print(f"[ERROR] Failed to extract {os.path.basename(filepath)}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during extraction: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Fully automated downloader and extractor for the AMASS dataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-u', '--username', type=str, required=True, help="Your AMASS account email.")
    parser.add_argument('-p', '--password', type=str, required=True, help="Your AMASS account password.")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="Directory to download and extract files to.")
    parser.add_argument('--smplh_only', action='store_true', help="Only download datasets from the 'smplh' category.", default=True)
    parser.add_argument('--specialize', type=str, help="Specialize the dataset to a specific category. e.g. 'CMU'", default=None)
    parser.add_argument('--cleanup', action='store_true', help="Delete .tar.bz2 archives after successful extraction.", default=True)
    parser.add_argument('--force', action='store_true', help="Force re-download of files, even if they exist.", default=False)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with requests.Session() as session:
        if not login(session, args.username, args.password):
            sys.exit(1)
            
        download_urls = get_download_urls(session, args.smplh_only, args.specialize)
        if not download_urls:
            sys.exit(1)
            
        for i, url in enumerate(download_urls, 1):
            print("-" * 60)
            print(f"Processing file {i}/{len(download_urls)}")
            
            downloaded_path = download_file(session, url, args.output_dir, args.force)
            if not downloaded_path:
                continue
            
            if extract_archive(downloaded_path, args.output_dir):
                if args.cleanup:
                    print(f"Cleaning up {os.path.basename(downloaded_path)}...")
                    os.remove(downloaded_path)
            else:
                print(f"Skipping cleanup for {os.path.basename(downloaded_path)} due to extraction error.")

    print("\n" + "="*60)
    print("All tasks completed.")
    print(f"AMASS dataset is ready in: {args.output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()