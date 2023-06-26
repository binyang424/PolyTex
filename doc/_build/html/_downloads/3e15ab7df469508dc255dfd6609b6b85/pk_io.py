"""
File input and output
=====================
All the functions related to file input and output were defined in
`polykriging.fileio` module. These functions allow users to read
and write files tailored for `polykriging`. The functions in this module
can be directly called by `polykriging.` + `function name`. For example,
to call the function choose_file, use `polykriging.fileio.choose_file`
or simply `polykriging.choose_file`.

For more information about these file formats, please refer to
"https://www.binyang.fun/polykriging/".

1. File selection
   In polykriging, two function were provided to facilitate file selection:
    1.1 polykriging.choose_file
    1.2 polykrigingchoose_directory
2. File reading and writing
"""

import numpy as np
import pandas as pd
import polykriging as pk

###############################################################################
# Select file and return the path
# -----------------------------------------------------------------------------
# The function `polykriging.choose_file()` allows users to select a file from
# a directory with a GUI. The function returns the path of the selected file.
# Users can also specify the file type and the title of the GUI window as below:
path = pk.choose_file(titl="Directory for file CoordinatesSorted file (.coo)", format=".coo")
coordinatesSorted = pk.pk_load(path)

###############################################################################
# Traverse and return a list of filenames in the directory
# -----------------------------------------------------------------------------
# The function `polykriging.filenames()` allows users to traverse a directory
# and return a list of filenames with a given extension.



label_row = pd.date_range("20130101", periods=6, freq="D", tz="UTC")
label_col = list("ABCD")
data = np.random.randn(6, 4)
df = pd.DataFrame(data, index=label_row, columns=label_col)

# save
pk.pk_save("./test_data/test.coo", df)

# load
df = pk.pk_load("./test_data/test.coo")

