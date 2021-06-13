#  NMTS 1 - tanks
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
import scipy.signal as sig
import qpsolvers

np.set_printoptions(linewidth=300)


# Parameters
g = 10  # m/s2
rho = 1000  # kg/m3
alph1 = 1e-6  # m3/(s*Pa)
alph2 = 1e-6  # m3/(s*Pa)
Nsteps = 20

# 171842,172118
a = 2
b = 8


S1 = 1+b  # m2
S2 = 0.5+a   # m2


# State model - continuous
A = np.array([[-alph1*rho*g/S1,             alph1*rho*g/S1],
              [alph1*rho*g/S2,  -(alph1 + alph2)*rho*g/S2]])

B = np.array([[1/S1],
              [0.0]])

C = np.array([0.0, alph2*rho*g])
D = np.array([0.0])
print(A)
print(B)
print(C)
print(D)


# Continuous model
G_c = sig.lti(A, B, C, D)
num, den = sig.ss2tf(A, B, C, D)  # Transfer function from state space
step_t_cont, step_resp_cont = G_c.step()  # Step response
print(f"Continuous transfer function numerator, denominator:  \n{num, den}")
print(f"Cont-time poles: {G_c.poles}")
print(f"Cont-time zeros: {G_c.zeros}")

# Get Ts = 1/15 * T_95
Ts = 1.0/15.0*step_t_cont[np.where(step_resp_cont>0.95)[0][0]]
print(f"Ts: {Ts}")

# Discrete model
model_d = sig.cont2discrete((A, B, C, D), dt=Ts)#, method='bilinear')  # Coversion from continuous model using billinear transform   method='bilinear'
A_d, B_d, C_d, D_d = model_d[0], model_d[1], model_d[2], model_d[3]
num_d, den_d = sig.ss2tf(A_d, B_d, C_d, D_d)  # Transfer function from state space
G_d = sig.dlti(A_d, B_d, C_d, D_d, dt=Ts)
step_t_disc, step_resp_disc = G_d.step(n=Nsteps)  # Step response
step_resp_disc = np.asarray(step_resp_disc).reshape(step_t_disc.shape)



# print(f"eigenvalues of A_d: \n{np.linalg.eigvals(A_d) }")
# print(f"\nG_d:\n{G_d}")
print(f"\nDiscrete State Space model:")
print(f"\nA_d:\n{A_d}")
print(f"\nB_d:\n{B_d}")
print(f"\nC_d:\n{C_d}")
print(f"\nD_d:\n{D_d}")
print(f"Discrete transfer function numerator, denominator:  \n{num_d, den_d }")
print(f"Discrete-time poles: {G_d.poles}")
print(f"Discrete-time zeros: {G_d.zeros}")

# Polynomials
Aq = np.zeros(den_d.shape[0] + 1)
Aq[:-1] = den_d  # den * 1
Aq[1:] -= den_d  # -den * q^-1
Bq = np.trim_zeros(num_d[0], trim='f')
# print(f"Aq: {Aq}")
# print(f"Bq: {Bq}")


# Predictive control

# Constrains
u_min = 0.0  # m3/s
u_max = 3.0  # m3/s
du_min = -2.0  # m3/s2
du_max = 2.0  # m3/s2

#  -------------------------------------
# Parameters
Hy = 9
Hu = 6
ro = 0.1
constr = False

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


n = Aq.shape[0] - 1
m = Bq.shape[0] - 1

y_ref = np.ones(Nsteps + Hy, dtype=np.float32)
y = np.zeros(Nsteps, dtype=np.float32)
u = np.zeros(Nsteps, dtype=np.float32)
du = np.zeros(Nsteps, dtype=np.float32)
x = np.zeros((Nsteps, A_d.shape[0]), dtype=np.float32)
t = np.arange(Nsteps)


for i in range(1, Nsteps):
    # Reference
    yr = y_ref[i+1:i+1+Hy]

    # prev, known y
    if i - n < 0:
        y_prev = np.zeros(n, dtype=np.float32)
        y_prev[-i-1:] = y[:i+1]
    else:
        y_prev = y[i+1-n:i+1]
    y_prev = np.flip(y_prev)

    # prev, known du
    if i-m <= 0:
        du_prev = np.zeros(m, dtype=np.float32)
        du_prev[-i:] = du[:i]
    else:
        du_prev = du[i-m:i]
    du_prev = np.flip(du_prev)

    # Find optimal du
    W = 2 * (H.T @ H + ro*np.eye(H.shape[1]))
    V = -2 * H.T @ (yr - P1 @ y_prev - P2 @ du_prev)
    if not constr:
        d_u_optimal = -inv(W) @ V
    else:
        Mt = np.tril(np.ones((Hu, Hu)))
        Eu = np.concatenate([Mt, -1 * Mt])
        Fu = np.concatenate([np.full(Hu, u_max - u[i - 1]), np.full(Hu, u_min + u[i - 1])])
        d_u_optimal = qpsolvers.solve_qp(W, V, Eu, Fu, lb=np.full(Hu, du_min), ub=np.full(Hu, du_max))
    du[i] = d_u_optimal[0]  # 1.T @ d_u_optimal

    u[i] = u[i - 1] + du[i]
    if i < Nsteps-1:  # Omit last step
        x[i+1] = A_d @ x[i] + np.transpose(B_d * u[i])
        y[i+1] = C_d @ x[i+1]

#  -------------------------------------


# Plots

# Controller dependent
#  -------------------------------------
# Controller - output
plt.figure(2)
plt.step(t, y)
plt.step(t, y_ref[:Nsteps])
plt.xlabel('Time [s]')
plt.ylabel('y[m^3/s]')
if constr:
    plt.title("Response with controller, with constraints")
else:
    plt.title("Response with controller, without constraints")
plt.legend(["y", "y_ref"])
plt.show()

# Controller - control signal
plt.figure(3)
plt.step(t, u)
plt.step(t, du)
plt.xlabel('Time [s]')
plt.ylabel('y[m^3/s]')
if constr:
    plt.title("Control signal with controller, with constraints")
else:
    plt.title("Control signal with controller, without constraints")
plt.legend(["u", "du"])
plt.show()
#  -------------------------------------


# Independent of controller
#  -------------------------------------
# Poles Continuous
# ax = plt.axes(xlim=(-.01, .01), ylim=(-.01, .01))
# for c in G_c.poles:
#     ax.plot(c.real, c.imag, "x")
# for c in G_c.zeros:
#     ax.plot(c.real, c.imag, "ro")
# ax.axhline(y=0, color='k')
# ax.axvline(x=0, color='k')
# ax.set_xlabel("Real")
# ax.set_ylabel("Imaginary")
# ax.set_title("s plane")
# ax.legend(["Poles"])
# plt.show()

# Poles Discrete
# ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-1.5,1.5))
# for c in G_d.poles:
#     ax.plot(c.real, c.imag, "x")
# for c in G_d.zeros:
#     ax.plot(c.real, c.imag, "ro")
# circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
# ax.add_patch(circ)
# ax.axhline(y=0, color='k')
# ax.axvline(x=0, color='k')
# ax.set_xlabel("Real")
# ax.set_ylabel("Imaginary")
# ax.set_title("z plane")
# ax.legend(["Poles"])
# plt.show()

# Continuous time without controller
# plt.figure(0)
# plt.plot(step_t_cont, step_resp_cont)
# plt.xlabel('Time [s]')
# plt.ylabel('y[m^3/s]')
# plt.title("Continuous time response without controller")
# plt.show()

# Discrete time without controller
# plt.figure(1)
# plt.step(step_t_disc, step_resp_disc)
# plt.xlabel('Time [s]')
# plt.ylabel('y[m^3/s]')
# plt.title("Discrete time response without controller")
# plt.show()
#  -------------------------------------

