import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import scipy as sc
import numpy as np
import sympy as sym
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sympy import init_printing
init_printing()

# Define the system parameters
m, k, c, t = sym.symbols('m, k, c, t')

# Define the main 
x = sym.Function('x')
lhs = m * x(t).diff(t,2) + k * x(t) + c * x(t).diff(t,1)
eq_main = sym.Eq(lhs, 0)
eq_main
eq_acc = sym.solve(eq_main, x(t).diff(t,2))[0]
eq_acc
sym.Eq(x(t).diff(t,2), sym.expand(eq_acc))
def SDOF_system(y, t, m, k, c):
    x, dx = y
    dydt = [dx,
           -c/m*dx - k/m*x]
    return dydt
m =1 # mass in kg
k= 200 # Spring Stiffnes N/m
c = 0.6# Dampening in Ns/m
y0 = [1.0, 0.0]
t = np.linspace(0, 3, 3000)
from scipy.integrate import odeint 
sol = odeint(SDOF_system, y0, t, args=(m, k, c))
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(t, sol[:, 0], 'b', label='x(t)')
#plt.plot(t, sol[:, 1], 'g', label='dx(t)/dt')
plt.legend(loc='best')
plt.xlabel('t (sec)')
plt.grid()

plt.show()

class FCN(nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fch(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    
def normalize(x):
    norm_x=(x-0)/(3-0)
    return norm_x
x=t
x=torch.tensor(x).reshape(len(x),1)
y=sol[:,0]
# slice out a small number of points from the LHS of the domain
x_data = x[0:3000:300]
x_data=torch.tensor(x_data).reshape(len(x_data),1)
y_data = y[0:3000:300]
y_data=torch.tensor(y_data).reshape(len(y_data),1)
ic_data=torch.zeros(1,1)
print(x_data.shape, y_data.shape)

def plot_result(x,y,x_data,y_data,yh,xp=None):
    "Pretty plot training results"
    plt.figure(figsize=(8,4))
    plt.plot(x,y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    plt.plot(x,yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0*torch.ones_like(xp), s=60, color="tab:green", alpha=0.4, 
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    plt.xlim(-0.05, 3)
    plt.ylim(-1.1, 1.1)
    plt.text(1.065,0.7,"Training step: %i"%(i+1),fontsize="xx-large",color="k")
    plt.axis("off")
    
x_physics = torch.linspace(0,3,3000).view(-1,1).requires_grad_(True)# sample locations over the problem domain
torch.manual_seed(123)
model = FCN(1,1,32,3)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
files = []
for i in range(50000):
    optimizer.zero_grad()
    
    # compute the "data loss"
   # yi=model(ic_data)
    yh = model(normalize(x_data).float())
    loss1 = torch.mean((yh-y_data)**2)# use mean squared error
    
    # compute the "physics loss"
    
    yhp = model(normalize(x_physics).float())
    dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]# computes dy/dx
    dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]# computes d^2y/dx^2
    physics = m*dx2 + c*dx + k*yhp       # computes the residual of the 1D harmonic oscillator differential equation
    loss2 = 1e-4*torch.mean(physics**2)
    #loss3=(torch.mean((yi-1)**2)+torch.mean(dx**2))
    # print("loss2=",loss2,  "loss3=  ", loss3)
    
    # backpropagate joint loss
    loss = loss1 + loss2# add two loss terms together
    #loss = loss1+ loss2+loss3
    loss.backward()
    optimizer.step()
    
    
    # plot the result as training progresses
    if (i+1) % 150 == 0: 
        
        yh = model(normalize(x).float()).detach()
        xp = x_physics.detach()
        
        plot_result(x,y,x_data,y_data,yh,xp)
        
        file = "plots/pinn_%.8i.png"%(i+1)
        # plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        files.append(file)
        
        if (i+1) % 100 == 0: plt.show()
        else: plt.close("all")
        print("loss1",loss1.item(),"loss2=",loss2.item(), "overall loss= ",loss.item())