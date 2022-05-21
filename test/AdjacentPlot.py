import matplotlib.pyplot as plt
import numpy as np

plt.style.use( 'seaborn-white')

t = np.arange(0.0, 2.0, 0.01)

s1 = np.sin(2 * np.pi * t)
s2 = np.exp(-t)
s3 = s1 * s2

fig, axs = plt.subplots(1, 3, sharey=True)
# Remove horizontal space between axes
fig.subplots_adjust(wspace=0)
fig.set_size_inches(4.5, 4)

# Plot each graph, and manually set the y tick values
axs[0].plot(t, s1)
##axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
axs[0].set_ylim(-1, 1)
axs[0].set_ylabel("Normalized distance")

axs[1].plot(t, s2)
axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
axs[1].set_ylim(0, 1)
axs[1].set_xlabel("Density")

axs[2].plot(t, s3)
axs[2].set_yticks(np.arange(-0.9, 1.0, 0.4))
axs[2].set_ylim(-1, 1)

plt.show()
