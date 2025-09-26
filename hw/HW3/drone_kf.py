import numpy
import math

default_dt = 1 / 200
default_var = 0.25

class DroneKF:
    def __init__(self, mass, thrust):
        self.m = mass
        self.T = thrust
        self.g = 9.81
        self.state = numpy.zeros((2,1))
        self.state_cov = numpy.identity(2) * default_var
        self.predicted_state = numpy.zeros((2,1))
        self.predicted_state_cov = numpy.identity(2) * default_var
        self.time = None

    def set_state(self, timestamp, position=0.0, velocity=0.0):
        self.time = timestamp
        self.state[0,0] = position
        self.state[1,0] = velocity
        self.state_cov = numpy.identity(2) * default_var

    def A_matrix(self, dt=default_dt):
        A = numpy.identity(2)
        A[0,1] = dt
        return A

    def B_matrix(self, dt=default_dt):
        B = numpy.zeros((2,1))
        B[0,0] = 0.5*dt**2
        B[1,0] = dt
        return B

    def C_matrix(self):
        C = numpy.zeros((1, 2))
        C[0,0] = 1
        return C

    def get_state(self):
        return self.state, self.state_cov

    def get_estimate(self):
        z = self.C_matrix() @ self.state
        z_cov = self.C_matrix() @ self.state_cov @ self.C_matrix().transpose()
        return z, z_cov

    def get_input(self, var=default_var):
        u_nominal = self.T / self.m - self.g
        u_var = var / self.m**2
        u = u_nominal + numpy.random.normal(0, math.sqrt(u_var))
        return u

    def peek_pos(self):
        return self.state[0, 0]

    def peek_vel(self):
        return self.state[1, 0]

    def _predict(self, timestamp, input, input_var=default_var):
        if self.time is None:
            raise RuntimeError('unintialized filter')
        delta_t = timestamp - self.time
        self.time = timestamp

        u = numpy.array([[input / self.m - self.g]])
        sigma_u = numpy.array([[input_var / self.m**2]])

        self.predicted_state = self.A_matrix(delta_t) @ \
            self.state + self.B_matrix(delta_t) @ u
        self.predicted_state_cov = \
            self.A_matrix(delta_t) @ self.state_cov @ \
            self.A_matrix(delta_t).transpose() + \
            self.B_matrix(delta_t) @ sigma_u @ \
            self.B_matrix(delta_t).transpose()

    def _skip_measure(self):
        self.state = self.predicted_state
        self.state_cov = numpy.identity(2) * default_var

    def _measure(self, measurement, measurement_var = default_var):
        sigma_z = numpy.array([[measurement_var]])
        z = numpy.array([[measurement]])
        SigmaXCT = self.predicted_state_cov @ self.C_matrix().transpose()
        CSigmaXCT = self.C_matrix() @ SigmaXCT
        K = SigmaXCT @ numpy.linalg.inv(CSigmaXCT + sigma_z)
        self.state = self.predicted_state + K @ \
            (z - self.C_matrix() @ self.predicted_state)
        self.state_cov = (numpy.identity(2) - K @ self.C_matrix()) @ \
            self.predicted_state_cov

    def advance_filter(self, timestamp, input, measurement, input_var, measurement_var):
        self._predict(timestamp, input, input_var)
        self._measure(measurement, measurement_var)

    def simulate_system(self, timestamp, input, input_var):
        self._predict(timestamp, input, input_var)
        self._skip_measure()
