import numpy as np

filename = "/workspace_QMRI/PROJECTS_DATA/2026_RECH_bruker_pulseq/pypulseq/shape/sech.inv"

x = []
y = []
xy = []

with open(filename, "r") as f:
    start = False

    for line in f:
        line = line.strip()

        if "##XYPOINTS=" in line:
            start = True
            continue

        if not start or not line:
            continue

        try:
            xi, yi = line.split(",")
            x.append(float(xi))
            y.append(float(yi))
            xy.append(float(xi))
            xy.append(float(yi))
        except ValueError:
            # fin des données ou ligne invalide
            break

x = np.array(x)
y = np.array(y)
xy = np.array(xy)

print(x[:10])
print(y[:10])