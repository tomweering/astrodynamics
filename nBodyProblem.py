# %%
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

#Just doing 2 body problem for now...
mi = 1                     #[kg]
mj = 10                    #[kg]
ri = np.array([0,5,0])     #[km]
rj = np.array([0,0,2])     #[km]
G = 1   	                #[m/s^2]

Q1 = np.empty(6)
Q1[:3] = ri
Q2 = np.empty(6)
Q2[:3] = rj

def length(r):
    result = np.sqrt(float(r[0])**2 + float(r[1])**2 + float(r[2])**2)
    return result

def fun1(Q, t):
    result = (1/mi)*(G*((mi*mj)/(length(Q2[:3]) - length(Q[:3]))**3)) * (Q2[:3] - Q[:3])
    return result

def fun2(Q, t):
    result = (1/mj)*(G*((mi*mj)/(length(Q1[:3]) - length(Q[:3]))**3)) * (Q1[:3] - Q[:3])
    return result

def RK4_1(dt, t0, x0):
    f1 = fun1(x0, t0)
    f2 = fun1(x0 + (dt/2)*f1, t0 + dt/2)
    f3 = fun1(x0 + (dt/2)*f2, t0 + dt/2)
    f4 = fun1(x0 + dt*f3, t0 + dt)
    x1 = x0 + (dt/6)*(f1 + 2*f2 + 2*f3 + f4)
    return x1

def RK4_2(dt, t0, x0):
    f1 = fun2(x0, t0)
    f2 = fun2(x0 + (dt/2)*f1, t0 + dt/2)
    f3 = fun2(x0 + (dt/2)*f2, t0 + dt/2)
    f4 = fun2(x0 + dt*f3, t0 + dt)
    x1 = x0 + (dt/6)*(f1 + 2*f2 + 2*f3 + f4)
    return x1


#setup the for loop and the step size
dt = 0.1
T = 10
t = np.arange(0,T,dt)

Y1 = np.zeros((3, len(t)))
Y2 = np.zeros((3, len(t)))

Y1[:, 0] = ri 
Y2[:, 0] = rj

for i in tqdm(range(len(t)-1)):
    Y1[:, i+1] = RK4_1(dt, t[i], Q1[:3])
    Y2[:, i+1] = RK4_2(dt, t[i], Q2[:3])

ax = plt.figure().add_subplot(projection="3d")
ax.plot(Y1[0, :], Y1[1, :], Y1[2, :], "b")
ax.plot(Y2[0, :], Y2[1, :], Y2[2, :], "b")
plt.show()
