# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:51:35 2022

@author: General
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May  1 22:28:14 2022

@author: General
"""
import scipy as sc
import numpy as np
import sympy as sym
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sympy import init_printing
init_printing()
import random
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
seconds = time.time()
m,k,c,gamma=1,200,0.6,160
def shm(u,t):
    x=u[0]
    y=u[1]
    dxdt=y
    dydt=-k*x-c*y-gamma*x**3
    return np.array([dxdt,dydt])
u0=[1,0]
total_time=10
tnp=20000       #total no. of points
ts=10            #total no. of segments
npes=int(tnp/total_time)    #no. of points each segment
t = np.linspace(0, total_time, tnp)
#t=np.linspace(0,10,20000)
sol=odeint(shm,u0,t)
x=sol[:,0]
y=sol[:,1]
plt.plot(t,x)
plt.show()
t=torch.tensor(t)
sol=torch.tensor(sol)
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
#nn_t=np.linspace(0, 1, 10000)
tol=1e-5
start=0
end=npes
pred_time_vector=(torch.zeros(npes,total_time).cuda()).data.squeeze()
for step in range(total_time):
    print("steps",step)
    
    #print("start",start)
    if step==0:
        lohfip=100  #length of initial high fidelity points
        nphfip=10   #number of high fidelity points
        x=t[start:end]
        pred_time_vector[:,step]=x
        x=x.cuda()
        x=x.reshape(len(x),1)
        y=sol[start:end,0].cuda()
        x_data = x[start:start+lohfip:nphfip].cuda()
        x_data=x_data.reshape(len(x_data),1)
        y_data=y[start:start+lohfip:nphfip].cuda()
        y_data=y_data.reshape(len(y_data),1)
    else:
        lonfip=200
        nonfip=10
        #x=t[start-200:end-200]
        x=t[start:end]
        pred_time_vector[:,step]=x 
        x=x.cuda()
        #print("xshape",x.shape)
        x=x.reshape(len(x),1)
        #y=sol[start-200:end-200,0]
        y=sol[start:end,0].cuda()
        x_data = x[0:0+lonfip:nonfip].cuda()
        x_data=x_data.reshape(len(x_data),1)
        y_data=yhp1[npes-lonfip:npes:nonfip].cuda()
        y_data=y_data.reshape(len(y_data),1)
    #pred_time_vector[:,step]=x
    print(x_data.shape, y_data.shape)
    def plot_result(x,y,x_data,y_data,yh,xp=None):
        "Pretty plot training results"
        plt.figure(figsize=(8,4))
        x=x.cpu()
        y=y.cpu()
        yh=yh.cpu()
        x_data=x_data.cpu()
        y_data=y_data.cpu()
        xp=xp.cpu()
        plt.plot(x.detach().numpy(),y.detach().numpy(), color="grey", linewidth=2, alpha=0.8, label="Exact solution")
        plt.plot(x.detach(),yh.detach().numpy(), color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
        plt.scatter(x_data.detach().numpy(), y_data[:,0], s=60, color="tab:orange", alpha=0.4, label='Training data')
        #print("xp",xp)
        if xp is not None:
            #print(xp.shape)
            plt.scatter(xp.detach().numpy(), -0*torch.ones_like(xp), s=60, color="tab:green", alpha=0.4, 
                    label='Physics loss training locations')
        l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
        plt.setp(l.get_texts(), color="k")
        plt.xlim(-0.05, 10)
        plt.ylim(-1.5, 1.5)
        plt.text(2.065,1.7,"Time step: %s"%(step+1),fontsize="xx-large",color="k")
        plt.text(1.065,0.7,"Training step: %i"%(i+1),fontsize="xx-large",color="k")
        plt.axis("off")
    x.requires_grad=True
    #x_physics = torch.linspace(step,step+1,2000).view(-1,1).requires_grad_(True)# sample locations over the problem domain
    x_physics=x.cuda()
    
    model = FCN(1,1,32,3).cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    alpha=[1e-5,1e-5,1e-5,1e-4,1e-4,1e-4,1e-4,1e-4,1e-3,1e-3]
    iteration=[55000,100000,1000000,1000000,1000000,1000000,1500000,2000000,2500000,2500000]
    for i in range(iteration[step]):
        optimizer.zero_grad()
    
    # compute the "data loss"
   # yi=model(ic_data)
        yh = model(x_data.float()).cuda()
        loss1 = torch.mean((yh-y_data)**2).cuda()                                # use mean squared error
    
    # compute the "physics loss"
    
        yhp = model(x_physics.float())
        yhp1=yhp
        yhp1=yhp1.clone().detach()
        dx  = torch.autograd.grad(yhp, x_physics, torch.ones_like(yhp), create_graph=True)[0]# computes dy/dx
        dx2 = torch.autograd.grad(dx,  x_physics, torch.ones_like(dx),  create_graph=True)[0]# computes d^2y/dx^2
        
        
        physics = m*dx2 + c*dx + k*yhp +gamma*yhp**3     # computes the residual of the 1D harmonic oscillator differential equation
        
        loss2 = alpha[step]*torch.mean(physics**2)

    #loss3=(torch.mean((yi-1)**2)+torch.mean(dx**2))
    # print("loss2=",loss2,  "loss3=  ", loss3)
    
    # backpropagate joint loss
        loss = loss1 + loss2# add two loss terms together
        
    #loss = loss1+ loss2+loss3
        loss.backward(retain_graph=(True))
        optimizer.step()
        #if loss1<5e-5:
           # alpha[step]=alpha[step]*10
        
        if loss<tol:
            yh = model(x.float()).cuda()
        
            xp = x_physics.cuda()
        
            plot_result(x,y,x_data,y_data,yh,xp)
            plt.show()
            break
    
    # plot the result as training progresses
        if (i+1) % 500 == 0: 
            #print("x",x.shape)
            yh = model(x.float()).cuda()
        
            xp = x_physics.cuda()
        
            plot_result(x,y,x_data,y_data,yh,xp)
        
            file = "plots/pinn_%.8i.png"%(i+1)
            #plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            #file.append(file)
        
            if (i+1) % 5000 == 0: plt.show()
            else: plt.close("all")
            print("epochs",i+1,"data loss",loss1.item(),"physics loss=",loss2.item(), "total loss= ",loss.item())
            
        
    torch.save(model.state_dict(), os.path.join('C:\\Nikhil Mahar\\sem2\\pinn\\codes\\harmonic-oscillator-pinn-main\\Model_hd', 'time_step-{}.pt'.format(step+1)))      
                                             #
    start=start+2000-200
    end=end+2000-200
print("Seconds since epoch =", seconds)	    
# =============================================================================
#     prediction 
# =============================================================================
pred_sol1=np.zeros([npes,total_time])
for i in range(total_time):
    with torch.no_grad():
        m1=model.load_state_dict(torch.load('C:\\Nikhil Mahar\\sem2\\pinn\\codes\\harmonic-oscillator-pinn-main\\Model_hd\\time_step-{}.pt'.format(i+1)))
        m1=model.eval()
        t_pred=pred_time_vector[:,i]
        t_pred=t_pred.reshape(len(t_pred),1)
        pred_sol=m1(t_pred)
        #t_pred=t.cpu().detach().numpy()
        pred_sol=pred_sol.cpu().detach().numpy()
        pred_sol1[:,i]=pred_sol[:,0]
        plt.plot(pred_time_vector[:,i].cpu().detach().numpy(),pred_sol1[:,i])
        #plt.plot(t, sol[:, 0], 'g', label='x(t)')
        #plt.show()
        
        



