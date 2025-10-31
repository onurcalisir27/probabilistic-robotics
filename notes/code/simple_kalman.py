import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_uncertainty_ellipse(ax, mu, Sigma, color, label, alpha=0.7):
    """Plot 1-sigma uncertainty ellipse"""
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(eigenvalues)

    ellipse = Ellipse(mu, width, height, angle=angle,
                     facecolor=color, edgecolor='black',
                     linewidth=2, alpha=alpha, label=label)
    ax.add_patch(ellipse)
    ax.plot(mu[0], mu[1], 'o', color='black', markersize=10, zorder=10)
    ax.plot(mu[0], mu[1], 'o', color=color, markersize=6, zorder=11)

    # Plot principal axes
    for i in range(2):
        axis = eigenvectors[:, i] * np.sqrt(eigenvalues[i])
        ax.arrow(mu[0], mu[1], axis[0], axis[1],
                head_width=0.4, head_length=0.4,
                fc=color, ec=color, alpha=0.6, linewidth=2)

# Parameters
C = np.array([[1, 0]])
Q = np.array([[10]])

# Before measurement
mu_before = np.array([0, 0])
Sigma_before = np.array([[20.25, 11.0],
                         [11.0, 5.0]])

# Measurement
z = 5

# Kalman Update
# Innovation
innovation = z - C @ mu_before
print(f"Innovation: ν = {innovation[0]:.3f}")

# Innovation covariance
S = C @ Sigma_before @ C.T + Q
print(f"Innovation covariance: S = {S[0,0]:.3f}")

# Kalman gain
K = Sigma_before @ C.T @ np.linalg.inv(S)
print(f"Kalman gain: K = [{K[0,0]:.4f}, {K[1,0]:.4f}]^T")

# Updated mean
mu_after = mu_before + K @ innovation
print(f"\nUpdated mean: μ = [{mu_after[0]:.3f}, {mu_after[1]:.3f}]^T")

# Updated covariance
Sigma_after = (np.eye(2) - K @ C) @ Sigma_before
print(f"Updated covariance:\n{Sigma_after}")

# Calculate uncertainty reduction
pos_reduction = (1 - Sigma_after[0,0]/Sigma_before[0,0]) * 100
vel_reduction = (1 - Sigma_after[1,1]/Sigma_before[1,1]) * 100
print(f"\nPosition variance reduced by: {pos_reduction:.1f}%")
print(f"Velocity variance reduced by: {vel_reduction:.1f}%")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Before measurement
ax1 = axes[0]
plot_uncertainty_ellipse(ax1, mu_before, Sigma_before,
                        'red', 'Before Measurement', alpha=0.3)
ax1.axhline(0, color='k', linewidth=0.5, alpha=0.3)
ax1.axvline(0, color='k', linewidth=0.5, alpha=0.3)
ax1.axvline(z, color='blue', linewidth=2, linestyle='--',
           label=f'Measurement z={z}', alpha=0.7)
ax1.set_xlabel('Position x', fontsize=14, fontweight='bold')
ax1.set_ylabel('Velocity ẋ', fontsize=14, fontweight='bold')
ax1.set_title('Before Measurement Update (t=5)', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12, loc='upper left')
ax1.set_xlim([-12, 12])
ax1.set_ylim([-8, 8])
ax1.set_aspect('equal')

# After measurement
ax2 = axes[1]
plot_uncertainty_ellipse(ax2, mu_before, Sigma_before,
                        'red', 'Before', alpha=0.15)
plot_uncertainty_ellipse(ax2, mu_after, Sigma_after,
                        'green', 'After Measurement', alpha=0.5)
ax2.axhline(0, color='k', linewidth=0.5, alpha=0.3)
ax2.axvline(0, color='k', linewidth=0.5, alpha=0.3)
ax2.axvline(z, color='blue', linewidth=2, linestyle='--',
           label=f'Measurement z={z}', alpha=0.7)

# Draw arrow showing update
ax2.annotate('', xy=mu_after, xytext=mu_before,
            arrowprops=dict(arrowstyle='->', lw=3, color='orange'))

ax2.set_xlabel('Position x', fontsize=14, fontweight='bold')
ax2.set_ylabel('Velocity ẋ', fontsize=14, fontweight='bold')
ax2.set_title('After Measurement Update (t=5)', fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12, loc='upper left')
ax2.set_xlim([-12, 12])
ax2.set_ylim([-8, 8])
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('kalman_measurement_update.png', dpi=150, bbox_inches='tight')
plt.show()

# Additional analysis plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot both ellipses overlaid
plot_uncertainty_ellipse(ax, mu_before, Sigma_before,
                        'red', 'Prior (Prediction)', alpha=0.3)
plot_uncertainty_ellipse(ax, mu_after, Sigma_after,
                        'green', 'Posterior (After Update)', alpha=0.5)

# Measurement constraint (vertical line)
y_range = np.linspace(-8, 8, 100)
ax.axvline(z, color='blue', linewidth=3, linestyle='--',
          label=f'Measurement z={z}', alpha=0.8)

# Shade measurement uncertainty
ax.fill_betweenx(y_range, z - np.sqrt(Q[0,0]), z + np.sqrt(Q[0,0]),
                 color='blue', alpha=0.2, label='Measurement Uncertainty (±1σ)')

ax.set_xlabel('Position x', fontsize=14, fontweight='bold')
ax.set_ylabel('Velocity ẋ', fontsize=14, fontweight='bold')
ax.set_title('Kalman Filter: Combining Prediction and Measurement',
            fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_xlim([-2, 10])
ax.set_ylim([-4, 6])
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('kalman_measurement_update_overlay.png', dpi=150, bbox_inches='tight')
plt.show()

# ''''''
#
# ### Output:
# Innovation: ν = 5.000
# Innovation covariance: S = 30.250
# Kalman gain: K = [0.6694, 0.3636]^T
#
# Updated mean: μ = [3.347, 1.818]^T
# Updated covariance:
# [[6.694 3.636]
#  [3.636 1.000]]
#
# Position variance reduced by: 66.9%
# Velocity variance reduced by: 80.0%
