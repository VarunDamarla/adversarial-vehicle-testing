import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

cwd = str(Path.cwd())
if "/src/adversarial_vehicle_testing/" not in cwd:
    cwd += "/src/adversarial_vehicle_testing/"
# Define parabola
def f(x): 
    return x**2

# Define parabola derivative
def slope(x): 
    return 2*x

# Define x data range for parabola
x = np.linspace(-5,5,100)

# Choose point to plot tangent line
x1 = -3
y1 = f(x1)

# Define tangent line
# y = m*(x - x1) + y1
def line(x, x1, y1):
    return slope(x1)*(x - x1) + y1

# Define x data range for tangent line
xrange = np.linspace(x1-1, x1+1, 10)

# Plot the figure
plt.figure()
plt.plot(x, f(x))
plt.scatter(x1, y1, color='C1', s=50)
plt.plot(xrange, line(xrange, x1, y1), 'C1--', linewidth = 2)
plt.savefig(cwd + "images/angle_vs_x.png")
