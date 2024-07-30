import matplotlib.pyplot as plt

# Angle with Perturbation vs Change in X wrt Center of Perturbation
plt.plot(xs, ys)
plt.xlabel("Change in X wrt Center of Perturbation")
plt.ylabel("Angle with Perturbation")
plt.title("Angle with Perturbation vs Change in X wrt Center of Perturbation")
for i in range(len(xs)):
    if int(xs[i]) == xs[i]:
        plt.plot(xs[i], ys[i], marker="o", markersize=5, color="red")
plt.savefig(cwd + "images/angle_vs_x.png")
