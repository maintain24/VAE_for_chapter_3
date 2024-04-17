"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create grid and multivariate normal
x = np.linspace(0, 1, 500)
y = np.linspace(0, 1, 500)
x, y = np.meshgrid(x, y)
pos = np.dstack((x, y))
mean = [0, 0]
covariance_matrix = [[1, 0], [0, 1]]
rv = np.random.multivariate_normal(mean, covariance_matrix, (500, 500))

# Make the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
z = rv[:,:,1]
ax.plot_surface(x, y, z, cmap='viridis')

# Annotate mean and variance
# ax.text2D(0.05, 0.95, "Mean (mu): 0", transform=ax.transAxes)
# ax.text2D(0.05, 0.90, "Variance (sigma squared): 1", transform=ax.transAxes)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Probability Density')

# Set background color
ax.set_facecolor('white')

plt.show()
"""

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the grid
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
x, y = np.meshgrid(x, y)

# Define the Gaussian function
def gaussian(x, y, x0, y0, xsig, ysig):
    return np.exp(-((x - x0) ** 2 / (2 * xsig ** 2) + (y - y0) ** 2 / (2 * ysig ** 2)))

# Mean and standard deviation for x and y
x0, y0 = 0, 0  # Mean
xsig, ysig = 1, 1  # Standard deviation

# Calculate the z values on the grid
z = gaussian(x, y, x0, y0, xsig, ysig)

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')

# Labeling the axes
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# Set the angle of the plot
ax.view_init(elev=30, azim=30)

plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the grid
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
x, y = np.meshgrid(x, y)

# Define the Gaussian function
def gaussian(x, y, x0, y0, xsig, ysig):
    return np.exp(-((x - x0) ** 2 / (2 * xsig ** 2) + (y - y0) ** 2 / (2 * ysig ** 2)))

# Mean and standard deviation for x and y
x0, y0 = 0, 0  # Mean
xsig, ysig = 1, 1  # Standard deviation

# Calculate the z values on the grid
z = gaussian(x, y, x0, y0, xsig, ysig)

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')

# # Labeling the axes
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# Set the angle of the plot
ax.view_init(elev=30, azim=30)

plt.show()
