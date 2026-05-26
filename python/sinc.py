import numpy as np
import matplotlib.pyplot as plt

# Données
x = np.linspace(-20, 20, 2000)
y = np.sinc(x / np.pi)  # numpy sinc = sin(pi x)/(pi x)



plt.figure()
plt.plot(x, y, linewidth=7, color='black')

plt.axis('off')
plt.gca().set_facecolor("none")  # fond transparent

plt.savefig("sinc_clean.svg", format="svg",
            bbox_inches='tight',
            pad_inches=0,
            transparent=True)

plt.close()


