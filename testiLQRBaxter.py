__author__ = 'fforget'

import numpy as np
from iLQRSolverBaxter import *
from costFunctionBaxter import *
from BaxterDynamicModel import *
import matplotlib.pyplot as pl
from random import uniform
import time


Xinit = np.matrix([ [0.0],
                    [0.0],
                    [0.0],
                    ])
Xinit = np.matrix([ [uniform(-10,10)],
                    [uniform(-10,10)],
                    [uniform(-10,10)],
                    ])
Xdes = np.matrix([  [uniform(-10,10)],
                    [uniform(-10,10)],
                    [uniform(-10,10)],
                    ])
XList = list()
UList = list()
XListtmp = list()
UListtmp = list()
timeList = list()

N = 20
 
"""Debug"""
traj = False
if (traj):
    M = 5
else:
    M = 1
trajList = list([Xdes,Xdes,Xdes,Xdes,Xdes])
dt = 1e-4

model = BaxterDynamicModel()
costFunction = CostFunctionBaxter()

print "X init_Original"
print Xinit
print "X destination_Original"
print Xdes

print " "
print " "
print "--------"
solver = iLQRSolver(model,costFunction)
for i in range(M):
    Xdes = trajList[i]
    initTime = time.time()
    XListtmp,UListtmp = solver.trajectoryOptimizer(Xinit,Xdes,N,dt, 500,1e-3)
    endTime = time.time() - initTime
    timeList.append(endTime/N)
    XList += XListtmp
    UList += UListtmp
    Xinit = XListtmp[N]


xList = list()
yList = list()
zList = list()
x_moveList = list()
y_moveList = list()
z_moveList = list()
for i in range(M*(N+1)):
    X = XList[i]
    xList.append(X[0].item())
    yList.append(X[1].item())
    zList.append(X[2].item())
for i in range(M*N):
    U = UList[i]
    x_moveList.append(U[0].item())
    y_moveList.append(U[1].item())
    z_moveList.append(U[2].item())
# print tauList
# print XList
# print UList
print timeList

print "X init"
print Xinit
print "X destination"
print Xdes

fig0 = pl.figure ()

ax0 = fig0.add_subplot ('221')
ax0.set_title('X')
ax0.plot ( range(M*(N+1)),xList)

bx0 = fig0.add_subplot ('222')
bx0.set_title('Y')
bx0.plot ( range(M*(N+1)),yList)

cx0 = fig0.add_subplot ('223')
cx0.set_title('Z')
cx0.plot ( range(M*(N+1)),zList)

ax0.grid()
bx0.grid()
cx0.grid()


fig1 = pl.figure()
ax1 = fig1.add_subplot ('221')
ax1.set_title('x_move')
ax1.plot ( range(M*N),x_moveList)

bx1 = fig1.add_subplot('222')
bx1.set_title('y_move')
bx1.plot( range(M*N), y_moveList)

cx1 = fig1.add_subplot('223')
cx1.set_title('z_move')
cx1.plot( range(M*N), z_moveList)

ax1.grid()
bx1.grid()
cx1.grid()

print costFunction.computeCostValue(N,XList,Xdes,UList)

pl.show()
