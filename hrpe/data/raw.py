import os
import requests
from zipfile import ZipFile


def download(verbose=True):
    """
    Downloads raw data and submission templates from links as provided on the link
    https://connecteddata.westernpower.co.uk/dataset/western-power-distribution-data-challenge-1-high-resolution-peak-estimation
    Weather data must be downloaded manually
    :param verbose if True, prints out helpful messages about
    """
    DATA_PACKAGE_JSON = "https://connecteddata.westernpower.co.uk/dataset/western-power-distribution-data-challenge-1-high-resolution-peak-estimation/datapackage.json"
    response = requests.get(DATA_PACKAGE_JSON)
    resource_urls = {
        r["name"].replace(" ", "_").lower(): r["url"]
        for r in response.json()["resources"]
        if r["url_type"] == "upload"
    }

    if verbose:
        print(f"Found {len(resource_urls)} resources to download")

    for name, url in resource_urls.items():
        if verbose:
            print(f"Downloading {name}")

        target_dir = os.path.join("data", "raw", name)
        target_zip = f"{target_dir}.zip"
        if os.path.isdir(target_dir):
            print(
                f"Warning: it looks like {name} has already been downloaded before, existing files will be overwritten"
            )

        resp = requests.get(url, stream=True)
        with open(target_zip, "wb") as h:
            h.write(resp.content)
        with ZipFile(target_zip) as z:
            z.extractall(target_dir)
        os.remove(target_zip)
