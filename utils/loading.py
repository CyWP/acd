import os
import urllib.request

try:
    from IPython import get_ipython

    if get_ipython() is not None:  # Check if running in a Jupyter Notebook
        from tqdm.notebook import tqdm
    else:
        raise ImportError  # Force fallback to standard tqdm
except ImportError:
    from tqdm import tqdm


def download_with_progress(url, folder, force_download=True):
    filename = get_filename_from_url(url)
    dest = os.path.join(folder, filename)
    if os.path.exists(dest) and not force_download:
        print(f"File {dest} already exists. Using cached file.")
        return dest
    with urllib.request.urlopen(url) as response, open(dest, "wb") as f, tqdm(
        total=int(response.info().get("Content-Length", 0)),
        unit="B",
        unit_scale=True,
        desc=f"Downloading to {dest}",  # Show file path in the progress bar
    ) as pbar:
        for chunk in iter(lambda: response.read(1024), b""):
            f.write(chunk)
            pbar.update(len(chunk))
    # Update the progress bar to show "Finished" status
    pbar.set_description(f"Finished downloading to {dest}")
    return dest


def get_filename_from_url(url):
    return url.split("/")[-1]
