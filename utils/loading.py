import os
import urllib.request

from tqdm import tqdm


def download_with_progress(url, folder, force_download=True):
    filename = get_filename_from_url(url)
    dest = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)
    if os.path.exists(dest) and not force_download:
        print(f"File {dest} already exists. Using cached file.")
        return dest
    with urllib.request.urlopen(url) as response, open(dest, "wb") as f, tqdm(
        total=int(response.info().get("Content-Length", 0)),
        unit="B",
        unit_scale=True,
        desc=f"Downloading to {dest}",
    ) as pbar:
        for chunk in iter(lambda: response.read(1024), b""):
            f.write(chunk)
            pbar.update(len(chunk))
    pbar.set_description(f"Finished downloading to {dest}")
    return dest


def get_filename_from_url(url):
    return url.split("/")[-1]
