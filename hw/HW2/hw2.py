#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
# index 0 means open
# index 1 means closed
#########################################################################
def bayes_update(bel, p_model, m_model, measurements):
    processed_beliefs = []
    for measurement in measurements:
        bel_bar = p_model @ bel
        unnormalized_posterior = m_model[measurement, :] * \
            bel_bar.flatten()
        posterior = unnormalized_posterior / sum(unnormalized_posterior)
        bel = np.array([posterior]).transpose()
        processed_beliefs.append(posterior[0])
    return processed_beliefs

#########################################################################

# rows are measurements, columns are priors
measurement_model = np.array([[0.9, 0.5],
                             [0.1, 0.5]])

# Prediction models
push_model = np.array([[1.0, 0.6],
                       [0.0, 0.4]])
no_action_model = np.array([[1.0, 0.0],
                            [0.0, 1.0]])

initial_bel = np.array([[0.5],
                        [0.5]])
measurements = [0, 0, 0, 0, 0, 1, 0, 0]
step = np.arange(0, len(measurements))

# Run the bayes filter for no action and push every time model
no_action_update = bayes_update(initial_bel, no_action_model, measurement_model, measurements)
push_update = bayes_update(initial_bel, push_model, measurement_model, measurements)

plt.plot(step, push_update, 'o-', label='With push')
plt.plot(step, no_action_update, 's-', label='No action')
plt.grid()
plt.xlabel('step')
plt.ylabel('belief(open)')
plt.ylim(0, 1.1)
plt.legend()
plt.show()
