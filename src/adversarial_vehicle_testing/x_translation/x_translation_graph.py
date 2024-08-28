import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

cwd = str(Path.cwd())
if "/src/adversarial_vehicle_testing/x_translation/" not in cwd:
    cwd += "/src/adversarial_vehicle_testing/x_translation/"

# Reading into CSV File
xs = []
ys = []
grads = []
with Path(cwd + "x_translation_data.csv").open("r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        grads.append(float(row[0]))
        xs.append(float(row[1]))
        ys.append(float(row[2]))

# Angle with Perturbation vs Change in X wrt Center of Perturbation
xticks = [-23, *list(np.arange(-20, 30, 5)), 28]
yticks = list(np.arange(-0.075, 0.150, 0.025))
plt.plot(xs, ys, linewidth=0.8)
plt.xticks(xticks)
plt.yticks(yticks)
plt.xlabel("Change in X wrt Center of Perturbation")
plt.ylabel("Angle with Perturbation")
plt.title("Angle with Perturbation vs Change in X wrt Center of Perturbation")
plt.savefig(cwd + "angle_vs_x.svg", dpi=300)
plt.close()

plt.plot(xs, grads, linewidth=0.8)
plt.xlabel("Change in X wrt Center of Perturbation")
plt.ylabel("Gradient of Angle with Perturbation")
plt.title("Gradient of Angle with Perturbation vs Change in X wrt Center of Perturbation", fontsize=11)
plt.savefig(cwd + "angle_grad_vs_x.svg", dpi=300)
plt.close()
