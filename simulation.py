#  NMTS 1 - tanks
import numpy as np
from matplotlib import pyplot as plt

def sim_step(x, u, A, B, C, dt=0.001):
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
A = np.array([[-alph1*rho*g*S1,             alph1*rho*g*S1],
              [alph1*rho*g*S2,  -(alph1 + alph2)*rho*g*S2]])

B = np.array([[S1],
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

plt.plot(H1)
plt.show()

