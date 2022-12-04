import os
import urllib


def example(data_name, outdir=r"./test_data/", overwrite=False):
    """
    Downloads sample datasets from the internet.

    Parameters
    ----------
    data_name : str
        Name of the dataset.
    outdir : str
        Output directory.
    overwrite : bool
        Overwrite existing files.
    """
    url = __data_list(data_name)
    fname = __download_file(url, outdir=outdir, overwrite=True)

    path = os.path.join(outdir, fname)

    return path


def __data_list(data_name):
    """
    This function contains a list of all available sample datasets. The url of the
    data can be retrieved by specifying the name of the dataset.

    Parameters
    ----------
    data_name : str
        Name of the dataset.

    Returns
    -------
    url : str
        URL of the dataset.
    """
    data_list = {
        "git": "https://raw.githubusercontent.com/binyang424/Git-for-beginners/master/git-cheat-sheet-education.pdf",
        "image": "https://www.binyang.fun/resource/test_data/binder_216_302.tif",
        "surface points": "https://www.binyang.fun/resource/test_data/binder_2.pcd", }

    return data_list[data_name]


def __get_filename(url):
    """
    Parses filename from the given url string.

    Parameters
    ----------
    url : str
        URL string.
    """
    if url.find('/'):
        return url.rsplit('/', 1)[1]


def __download_file(url, outdir=r"./test_data/", overwrite=True):
    """
    Downloads file from the given url and saves it to the given directory.

    Parameters
    ----------
    url : str
        URL string.
    outdir : str
        Output directory.
    """
    # Create folder if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Parse filename
    fname = __get_filename(url)
    outfp = os.path.join(outdir, fname)
    # Download the file if it does not exist already
    if not os.path.exists(outfp):
        # print("Downloading", fname)
        r = urllib.request.urlretrieve(url, outfp)
    else:
        if overwrite:
            print("Overwriting", fname)
            r = urllib.request.urlretrieve(url, outfp)
        else:
            print("File already exists:", fname)

    return fname


if __name__ == "__main__":

    # File locations
    url_list = ["https://www.binyang.fun/resource/Semivariogram_1D_Porosity.xlsx",
                "https://www.binyang.fun/resource/LICENSE",
                "https://www.binyang.fun/resource/Thermoset_FRP_Manufacturing_Fundamentals.pdf",
                "https://raw.githubusercontent.com/binyang424/Git-for-beginners/master/git-cheat-sheet-education.pdf"
                ]

    # Download files
    for url in url_list:
        __download_file(url)
