import os
import urllib.request as request


def example(data_name, outdir=r"./test_data/", overwrite=False):
    """
    Downloads sample datasets from the internet.

    Parameters
    ----------
    data_name : str
        Name of the dataset. Use ``data_name="all"`` to see a list of all
        available datasets.
    outdir : str
        Output directory.
    overwrite : bool
        Overwrite existing files.
    """
    if data_name == "all":
        __data_list(data_name)
    else:
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
        "image": "https://www.binyang.fun/resource/test_data/Example_implicit_dataset.tif",
        "surface points": "https://github.com/binyang424/PolyTex/raw/master/test/sample_data/Example_explicit_dataset.zip",
        "cross-section": "https://www.binyang.fun/resource/test_data/boundary_of_cross-section.vtk",
        "sorted coordinate": "https://github.com/binyang424/PolyTex/raw/master/test/sample_data/vf57_weft_0.coo",
        "case template": "https://github.com/binyang424/PolyTex/raw/master/test/sample_data/CaseTemplate.zip",}
    dataset_info = {
        "image": "An image sequence of two  binder tows obtained by Micro CT scanner.",
        "surface points": "The surface points of a binder tow.",
        "cross-section": "The cross-section of the binder tow.",
        "sorted coordinate": "This file contains the sorted coordinates of a tow and the corresponding parameters.",
        "case template": "A template case for OpenFOAM simulation. Use Textile.case_prepare() to setup the case."}

    if data_name == "all":
        print("Available datasets:" + "\n")
        for key in data_list.keys():
            print(key, ":", dataset_info[key])
    else:
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

    Returns
    -------
    fname : str
        Filename.

    Examples
    --------
    >>> from polytex import __download_file
    >>> # File locations
    >>> url_list = ["https://www.binyang.fun/resource/Semivariogram_1D_Porosity.xlsx", \
                "https://www.binyang.fun/resource/LICENSE"]
    >>> # Download files
    >>> for url in url_list:
    >>>    __download_file(url)
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
        r = request.urlretrieve(url, outfp)
    else:
        if overwrite:
            print("Overwriting", fname)
            r = request.urlretrieve(url, outfp)
        else:
            print("File already exists:", fname)

    return fname