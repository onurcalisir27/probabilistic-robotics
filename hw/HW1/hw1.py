import random
import numpy as np
import matplotlib.pyplot as plt
import math

print("------------------Problem 1------------------------------- \n")
# Thinking of a circle(r=1) and square(a=2) centered at the origin
# within the 1st quadrant[0,1), we can compare how many points fall inside the
# circle vs outside the circle (inside the square)
# Theoretically:
# points_inside_circle / points_inside_square = Area(circ) / Area(square)
# pic / pis = pi*1^2 / 2*2 = pi / 4
# pi = 4 * pic / pis

points_inside_circle = 0
sample_size = 100000000
for i in range(sample_size):
    x = random.random()
    y = random.random()
    dist = x**2 + y**2
    if dist <= 1:
      points_inside_circle += 1

estimate_of_pi = 4 * points_inside_circle / sample_size
print(f"Estimated Pi value, with {sample_size} samples, is = {estimate_of_pi} \n")
# print(" ")

print("------------------Problem 4-------------------------------\n")
rot_angle = np.radians(30)
U = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
             [np.sin(rot_angle), np.cos(rot_angle)]])
L = np.array([[1.0, 0.0], [0.0, 0.25]])

Cov = U @ L @ U.T
detCov = np.linalg.det(Cov)
Covinv = np.linalg.inv(Cov)
term = 1 / (2*np.pi*np.sqrt(detCov))

print(f"Covariance Matrix:\n{Cov}")
print(f"Determinant of Covariance Matrix: {detCov}")
print(f"Inverse Covariance Matrix:\n{Covinv}")
print(f"Heading Term: {term}")

mean = np.array([3.0, 4.0])
num_samples = 1500
samples = np.random.multivariate_normal(mean, Cov, num_samples)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)

# Generate points on the unit circle to form the ellipse
t = np.linspace(0, 2*np.pi, 100)
ellipse_points = np.array([np.cos(t), np.sin(t)])

eigenvalues = np.diag(L)
scaled_ellipse = 3 * np.sqrt(eigenvalues)[:, np.newaxis] * ellipse_points
rotated_ellipse = np.dot(U, scaled_ellipse)

# Plot the uncertainty ellipse
plt.plot(mean[0] + rotated_ellipse[0, :], mean[1] + rotated_ellipse[1, :], color='red')

# Plot eigenvectors as arrows
plt.quiver(mean[0], mean[1], U[0, 0], U[1, 0], angles='xy', scale_units='x', scale=0.3, color='blue')
plt.quiver(mean[0], mean[1], U[0, 1], U[1, 1], angles='xy', scale_units='x', scale=0.3, color='blue')

# Draw x and y axes
plt.axhline(0, color='green', linewidth=1.5)
plt.axvline(0, color='green', linewidth=1.5)
plt.xlabel(r'$x_{1}$')
plt.ylabel(r'$x_{2}$')
plt.xlim(-5, 10)
plt.ylim(-5, 10)
plt.axis('equal')
plt.grid()
print(" ")

print("------------------Problem 5-------------------------------\n")
# Function to simulate the scenario
def problem3(children):
  x = random.random()
  if x < 0.5:
    # We got a boy
    return children + 1
  return  problem3(children + 1)

sample = 10000000
num_children = 0
for i in range(sample):
  children = 0
  num_children += problem3(children)

total_children_per_family = num_children / sample
print(f"Out of {sample} samples, total number of children per family is {total_children_per_family}")

######################################################
# Show the plot
plt.show()
