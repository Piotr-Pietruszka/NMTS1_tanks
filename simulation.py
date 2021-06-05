#  NMTS 1 - tanks
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz

np.set_printoptions(linewidth=300)


def sim_step(x, u, A, B, C, dt=0.001):
    """
    Simulate one time step for given system, with given state-space model
    :param x: state at time t
    :param u: input at time t
    :param A: state transition matrix
    :param B: input matrix
    :param C: output matrix
    :param dt: time-step size
    :return: state at time t+dt, output at time t
    """
    x_next = x + A@x + B*u
    y = C @ x

    return x_next, y


# Parameters
g = 10  # m/s2
rho = 1000  # kg/m3
alph1 = 1e-6  # m3/(s*Pa)
alph2 = 1e-6  # m3/(s*Pa)

# ???
a = 1
b = 1

S1 = 1+b  # m2
S2 = 0.5+a   # m2


# State model
A = np.array([[-alph1*rho*g/S1,             alph1*rho*g/S1],
              [alph1*rho*g/S2,  -(alph1 + alph2)*rho*g/S2]])

B = np.array([[1/S1],
              [0]])

C = [0, alph2*rho*g]

print(A)
print(B)
print(C)


# Initial conditions
H1_0 = 0
H2_0 = 0

x = np.array([[H1_0],
                [H2_0]])

# Main time loop
H1 = []
H2 = []
Nsteps = 100
for t in range(Nsteps):
    x, y = sim_step(x, u=1, A=A, B=B, C=C, dt=0.001)
    H1.append(x[0])
    H2.append(x[1])

# plt.plot(H1)
# plt.show()



# Predictive control
Aq = np.array([1., -1.83345819, 0.88324525, -0.04978707])
Bq = np.array([0.15645635, 0.05987255])



# Parameters
N = 30
Hy = 10
Hu = 8

# C_A, C_B, H_A, H_B matrices

first_c = np.zeros(Hy)  # first column
first_c[:Aq.shape[0]] = Aq
C_A = toeplitz(first_c, np.zeros(Hy))  # toeplitz matrix - first row - all 0 (except first element)

first_c = np.zeros(Hy)  # first column
first_c[:Bq.shape[0]] = Bq
C_B = toeplitz(first_c, np.zeros(Hu))  # toeplitz matrix - first row - all 0 (except first element)

H_A = np.zeros((Hy, Aq.shape[0] - 1), dtype=np.float32)
for i in range(Aq.shape[0] - 1):
    H_A[i, :H_A.shape[1]-i] = Aq[i+1:]

H_B = np.zeros((Hy, Bq.shape[0] - 1), dtype=np.float32)
for i in range(Bq.shape[0] - 1):
    H_B[i, :H_B.shape[1] - i] = Bq[i + 1:]


# H, P1, P2 matrices
H = inv(C_A) @ C_B
P1 = -1 * inv(C_A) @ H_A
P2 = inv(C_A) @ H_B



print()
print(f"\nAq:\n{Aq}")
print(f"\nBq:\n{Bq}")
print(f"\nP1:\n{P1}")
print(f"\nP2:\n{P2}")
print(f"\nC_A:\n{C_A}")
print(f"\nC_B:\n{C_B}")
print(f"\nH_A:\n{H_A}")
print(f"\nH_B:\n{H_B}")












