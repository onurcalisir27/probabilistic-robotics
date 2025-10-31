import numpy as np

# Measurement z tells me how much candy i have left
# x_t+1 = x_t - u_t
bel = [50]
z = [48, 48]
# P(x(t) | x(t-1), u(t))
# P(z(t) | x(t))

prediction_mdl = np.array([[0.6], [0.3], [0.1]])

# 0.6 perceent chance we have x=49
# 0.3 percent change we have x=48
# 0.1 percent change we have x=47

# we get measurement 48

# if we had u=1, x=49: 0.6 * 0.2 = 0.12
# if we had u=2, x=48: 0.3 * 0.6 = 0.18
# if we had u=3, x=47: 0.1 * 0.2 = 0.02
unnormalized_posterior = np.array([[0.12], [0.18], [0.02]])
posterior = unnormalized_posterior / sum(unnormalized_posterior)

# [0.375  0.5625 0.0625] means
# [49 48 47]

"""
if we had x_t-1 = 49, u=1: x=48 0.375 * 0.6
                      u=2: x=47 0.375 * 0.3
                      u=3: x=46 0.375 * 0.1

if we had x_t-1 = 48, u=1: x=47 0.5625 * 0.6
                      u=2: x=46 0.5625 * 0.3
                      u=3: x=45 0.5625 * 0.1

if we had x_t-1 = 47, u=1: x=46 0.0625 * 0.6
                      u=2: x=45 0.0625 * 0.3
                      u=3: x=44 0.0625 * 0.1

    we got measurement z=48 again

Which means we could have gotten P(z=48 | x=48): (0.375 * 0.6) * 0.6
                                 P(z=48 | x=47): (0.375 * 0.3) * 0.2 + (0.5625 * 0.6) * 0.2

"""
# Take action
bel_48 = 0.6 * 0.375
bel_47 = 0.375 * 0.3 + 0.5626 * 0.6
bel_46 = 0.375 * 0.1 + 0.5625 * 0.3 + 0.0625 * 0.6
bel_45 = 0.5625 * 0.1 + 0.0625 * 0.3
bel_44 = 0.0625 * 0.1
bel_bar = np.array([[bel_48], [bel_47], [bel_46], [bel_45], [bel_44]])
unnormalized_posterior = np.array([[bel_48], [bel_47], [bel_46], [bel_45], [bel_44]])

measurement_model = np.array([0.6, 0.2, 0.0, 0.0, 0.0])
bel_bar = np.array([0.225, 0.450, 0.244, 0.075, 0.006])
unnormalized_posterior = bel_bar * measurement_model
posterior = unnormalized_posterior / sum(unnormalized_posterior)

print(f"{posterior}")
# bel = [z=x+1, z=x, z=x-1]
measurement_mdl = np.array([[0.2, 0.6, 0.2]])

After visit 1 (z=48): bel(x(1)) = [0.375, 0.5625, 0.0625] for x ∈ {49, 48, 47}
After visit 2 (z=48): bel(x(2)) = [0.6, 0.4, 0.0, 0.0, 0.0] for x ∈ {48, 47, 46, 45, 44}
