#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def bayes_update(bel, p_model, m_model, measurements):
    processed_beliefs = []
    for measurement in measurements:
        bel_bar = p_model @ bel
        unnormalized_posterior = m_model[measurement, :] * bel_bar.flatten()
        posterior = unnormalized_posterior / sum(unnormalized_posterior)
        bel = np.array([posterior]).transpose()
        processed_beliefs.append(posterior)
    return processed_beliefs

# Measurement model
measurement_model = np.array([[0.6, 0.3, 0.0],  # z=0 (sunny)
                             [0.4, 0.7, 0.0],   # z=1 (cloudy)
                             [0.0, 0.0, 1.0]])  # z=2 (rainy)

# Transition model
transition_model = np.array([[0.8, 0.4, 0.2],  # next=sunny
                             [0.2, 0.4, 0.6],  # next=cloudy
                             [0.0, 0.2, 0.2]]) # next=rainy

# Initial belief:
b_x0 = np.array([[1.0],   # sunny
                 [0.0],   # cloudy
                 [0.0]])  # rainy

# Measurements: cloudy, cloudy, rainy, sunny
z = [1, 1, 2, 0]

# Run Bayes filter
update = bayes_update(b_x0, transition_model, measurement_model, z)

# Extract probabilities per state
sunny = [u[0] for u in update]
cloudy = [u[1] for u in update]
rainy = [u[2] for u in update]

# Answer to problem (a): Day 5 (after 4 measurements)
print(f"\nDay 5 belief after observing [cloudy, cloudy, rainy, sunny]:")
print(f"P(sunny) = {update[3][0]:.4f}")
print(f"P(cloudy) = {update[3][1]:.4f}")
print(f"P(rainy) = {update[3][2]:.4f}")

# Plot
step = np.arange(1, len(z) + 1)
plt.plot(step, sunny, 'yo-', linewidth=2, markersize=8, label='Sunny')
plt.plot(step, cloudy, 'go-', linewidth=2, markersize=8, label='Cloudy')
plt.plot(step, rainy, 'bo-', linewidth=2, markersize=8, label='Rainy')
plt.grid()
plt.xlabel('Day (after Day 1)')
plt.ylabel('Belief')
plt.ylim(0, 1.1)
plt.title('Weather Belief Evolution')
plt.legend()
plt.show()

