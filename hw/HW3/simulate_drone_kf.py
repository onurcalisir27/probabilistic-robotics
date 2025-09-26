from drone_kf import DroneKF
import matplotlib.pyplot as plt
import numpy
import math
import random

m = 0.25
T = 2.7
thrust_var = 0.25
g = 9.81
time_step = 1 / 200
sim_time = 5
total_steps = int(sim_time / time_step)

bot_model = DroneKF(m, T)
bot_model.set_state(0)

estimator = DroneKF(m, T)
estimator.set_state(0)

mismatched_estimator = DroneKF(1.1*m, T)
mismatched_estimator.set_state(0)

all_t = [i * time_step for i in range(1, total_steps)]
noisy_thrust = [T + numpy.random.normal(0, math.sqrt(thrust_var)) for i in all_t]

min_var_z = 0.01
max_var_z = 0.5

h_ground_truth = []
h_sensor = []
h_estimate = []
h_mismatched_estimate = []

def generate_measurement(hm):
    var_zh = random.uniform(min_var_z, max_var_z)
    zh = numpy.random.normal(hm, math.sqrt(var_zh))
    return zh, var_zh

def calculate_error(gt, diff):
    error = [x-y for x, y in zip(gt, diff)]
    return error

for i in range(len(all_t)):

    input = noisy_thrust[i]
    bot_model.simulate_system(all_t[i], input, thrust_var)
    hgt = bot_model.peek_pos()
    h_ground_truth.append(hgt)

    zh, var_zh = generate_measurement(hgt)
    h_sensor.append(zh)

    estimator.advance_filter(all_t[i], input, zh, thrust_var, var_zh)
    h_est, h_est_cov = estimator.get_estimate()
    h_estimate.append(h_est[0])

    mismatched_estimator.advance_filter(all_t[i], input, zh, thrust_var, var_zh)
    h_mismatched_est, h_mismatched_est_cov = mismatched_estimator.get_estimate()
    h_mismatched_estimate.append(h_mismatched_est[0])

plt.figure(1)
plt.title('Drone Altitude in 4 Different Scenarios')
plt.plot(all_t, h_ground_truth, c='k', linewidth=3, label='Ground Truth Altitude')
plt.plot(all_t, h_sensor, c='y', alpha=0.3, label='Sensor Altitude')
plt.plot(all_t, h_estimate, c='r', linewidth=2, linestyle='dashed', label='Estimated Altitude')
plt.plot(all_t, h_mismatched_estimate, c='b', linewidth=2, linestyle='dashed', label='Model Mismatch Estimated Altitude')
plt.ylabel('Altitude H [m]')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()

sensor_error = calculate_error(h_ground_truth, h_sensor)
estimate_error = calculate_error(h_ground_truth, h_estimate)
mismatched_estimate_error = calculate_error(h_ground_truth, h_mismatched_estimate)

plt.figure(2)
plt.title('Drone Altitude Error from Simulation Ground Truth')
plt.plot(all_t, sensor_error, c='y', alpha=0.3, label='Sensor Measurement Error')
plt.plot(all_t, estimate_error, c='r', label='Kalman Filter Estimation Error')
plt.plot(all_t, mismatched_estimate_error, c='b', label='Mismatched Model KF Estimation Error')
plt.ylabel('Altitude Error [m]')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()

plt.show()
