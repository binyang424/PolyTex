"""
Fiber volume fraction of tow
====================

Test

"""

import polykriging as pk
import matplotlib.pyplot as plt
import re, pyLCM

path = pk.fileio.choose_directory(titl="geo files")
cwd = pk.fileio.cwd_chdir(path)  # ./transformation

filenames = pk.fileio.filenames(path, "geo")

labels = [int(re.findall(r'\d+', filename)[0]) for filename in filenames]
# sort the filenames according to the labels
filenames = [filename for _, filename in sorted(zip(labels, filenames))]

for filename in filenames[:5]:
    geo = pk.pk_load(filename)

    if "warp" in filename:
        tex = 2200  # tex: 1 tex = 1 g/km
    elif "weft" in filename:
        tex = 1100
    else:
        tex = 275  # binder

    print(filename)
    fiber_area = pyLCM.utility.tex_to_area(tex, density_fiber=2550)  # mm^2
    vf = fiber_area / geo["Area"]
    # print the maximum and minimum fiber volume fraction
    print("The maximun and the minimum fiber volume fraction: %.2f and %.2f"
          % (vf.min(), vf.max()))

    # plot the fiber volume fraction.
    try:
        ax.plot(geo["centroidZ"], vf, label=filename)
    except NameError:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ax.set_xlabel("y (mm)")
ax.set_ylabel("fiber volume fraction")
ax.set_ylim(0.5, 1)
ax.legend()
plt.show()
