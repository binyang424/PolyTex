import polykriging as pk
import numpy as np
import matplotlib.pyplot as plt
import re, pyLCM

path = pk.fileio.choose_directory(titl="geo files")
cwd = pk.fileio.cwd_chdir(path)  # ./transformation

filenames = pk.fileio.filenames(path, "geo")

labels = [int(re.findall(r'\d+', filename)[0]) for filename in filenames]
# sort the filenames according to the labels
filenames = [filename for _, filename in sorted(zip(labels, filenames))]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for filename in filenames[:2]:
    geo = pk.pk_load(filename)
    tex = 1100  # tex: 1 tex = 1 g/km

    print(filename)
    fiber_area = pyLCM.utility.tex_to_area(tex, density_fiber=2550)  # mm^2
    vf = fiber_area / geo["Area"]
    # print the maximum and minimum fiber volume fraction
    print("The maximun and the minimum fiber volume fraction: %.2f and %.2f"
          % (vf.min(), vf.max()))

    # plot the fiber volume fraction.
    ax.plot(geo["centroidY"], vf, label=filename)

ax.set_xlabel("y (mm)")
ax.set_ylabel("fiber volume fraction")
ax.set_ylim(0.5, 1)
ax.legend()
plt.show()