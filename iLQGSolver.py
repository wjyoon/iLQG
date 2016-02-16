# Plotting
import matplotlib.pyplot as plt

# Numerics
import numpy as np
import numpy.linalg

#Cost function and Dynamic Model
from costFunction import *
from dynamicModel import *
import time

'''

Codes originally from https://github.com/flforget/DDP
Algorithm based on Synthesis and Stabilization of Complex Behaviors through Online Trajcetory Optimization


'''


# Algorithms from iterative Linear Quadratic Gaussian(iLQG) trajectory optimizer, variant of the classic DDP
###############################################################################################################################
# Steps for solving Trajectory Optimizer

# 0.   Initiallization x, u, l(cost function), f, Q, V
# 1.   Derivatives
#      Given a nominal sequence (x, u, i) computes first and second derivatives of l and f, which will be each, Jacobian and Hessian
# 2-1. BackWard Pass
#      Iterating equations related to Q, K and V for decreasing i = N-1, ... 1.
# 2-2. Condition Hold
#      If non-PD(Positive Definite) Q_uu is encountered, increase mu, and restart the backward pass or, decrease mu if successful.
# 3-1. Forward Pass
#      set alpha = 1. Iterate the new controller u and the new nominal sequence (x, u, i)
# 3-2. Convergence
#      A condition for convergence is satisfied, done, or decrease alpha and restart the forward pass.
###############################################################################################################################


class iLQRSolver:
    def __init__(self, model, costFunction):
        
        # Initializations 
        # Note
        # model, costFunction, X, U, F, derivatives 
        self.model = model
        self.costFunction = costFunction
        self.Xinit = np.zeros((model.stateNumber, 1))
        self.Xdes = np.zeros((model.stateNumber,1))
        self.T = 10
        self.dt = 1e-4
        self.iterMax = 20
        self.stopCriteria = 1e-3 # Could be fixed
        
        self.changeAmount = 0.0
        self.flag_BackwardPass = False
        
        self.X = np.zeros((self.model.stateNumber, 1)) # (x, y, z) coordinates, stateNumber = 3
        self.U = np.zeros((self.model.commandNumber, 1)) # commandNumber = 6
        self.nextX = np.zeros((self.model.stateNumber, 1))
        self.XList = []
        self.UList = []
        self.nextXList = []
        self.nextUList = []
        
        # Parameters for running conditions
        self.alpha = 1.0 #backtracking line-search parameter 0 < alpha <= 1
        self.alphaList = [1.0, 0.8, 0.6, 0.4, 0.2] # Could be fixed
        self.mu = 0.0 # the roloe of a Lavenberg-Marquardt parameter
        self.muEye = self.mu*np.eye(self.model.stateNumber, dtype = float)
        
        self.zerosCommand = np.zeros((self.model.commandNumber, 1))
        
        self.kList = []
        self.KList = []
        self.k = np.zeros((self.model.commandNumber, 1))
        self.K = np.zeros((self.model.commandNumber, self.model.stateNumber))
        
        # Derivatives of Q and V functions: First, and Second Derivatives
        self.Qx = np.zeros((self.model.stateNumber, 1))
        self.Qu = np.zeros((self.model.commandNumber, 1))
        self.Qxx = np.zeros((self.model.stateNumber, self.model.stateNumber))
        self.Quu = np.zeros((self.model.commandNumber, self.model.commandNumber))
        self.Quu_inv = np.zeros((self.model.commandNumber, self.model.commandNumber))
        self.Qux = np.zeros((self.model.commandNumber, self.model.stateNumber))  # 6x3
        self.nextVx = np.zeros((self.model.stateNumber, 1))
        self.nextVxx = np.zeros((self.model.stateNumber, self.model.stateNumber))
    
    def trajectoryOptimizer(self, Xinit, Xdes, T, dt, iterMax = 20, stopCrit = 1e-3):
        #initialization
        self.Xinit = Xinit
        self.Xdes = Xdes
        self.T = T
        self.dt = dt
        self.iterMax = iterMax
        self.stopCrit = stopCrit
        
        self.initTrajectory()
        for iter in range(self.iterMax):
            self.backwardLoop()
            self.forwardLoop()
            self.XList = self.nextXList
            self.UList = self.nextUList
            if(self.changeAmount < self.stopCrit): 
                # condition for ending the iterations
                # changeAmount is updated in forwardLoop()
                break
        return self.XList, self.UList
        
    def initTrajectory(self):
        self.XList = [self.Xinit] # a list of numpy arrays
        self.UList = [self.zerosCommand for i in range(self.T)]
        for i in range(self.T):
            self.model.computeNextState(self.dt, self.XList[i], self.UList[i])  ## To be fixed
            self.XList.append(self.model.nextX)                                 ## To be fixed
        return 0
    
    def backwardLoop(self):
        self.kList = []
        self.KList = []
        self.costFunction.computeFinalCostDeriv(self.XList[self.T], self.Xdes) ## To be checked
        self.nextVx = self.costFunction.lx
        self.nextVxx = self.costFunction.lxx
        self.mu = 0.0001
        self.flag_BackwardPass = False
        while(self.flag_BackwardPass == 0):
            self.muEye = self.mu * np.eye(self.nextVxx.shape[0], dtype = float)
            for i in range(self.T-1, -1, -1):
                self.X = self.XList[i]
                self.U = self.UList[i]
                
                # Get Derivatives using the given sequence
                self.model.computeAllModelDeriv(self.dt, self.X, self.U) # derivatives of f
                self.costFunction.computeAllCostDeriv(self.X, self.Xdes, self.U) # derivative of l
                
                # Updates derivatives of the Q function
                self.Qx = self.costFunction.lx + np.dot(self.model.fx.T,self.nextVx)
                self.Qu = self.costFunction.lu + np.dot(self.model.fu.T,self.nextVx)
                self.Qxx = self.costFunction.lxx + np.dot(np.dot(self.model.fx.T,self.nextVxx),self.model.fx)
                self.Quu = self.costFunction.luu + np.dot(np.dot(self.model.fu.T,(self.nextVxx+self.muEye)),self.model.fu)
                self.Qux = self.costFunction.lux + np.dot(np.dot(self.model.fu.T,(self.nextVxx+self.muEye)),self.model.fx)
                    
                for j in range(self.model.stateNumber):
                    self.Qxx += np.dot(self.nextVx[j].item(),self.model.fxx[j])
                    self.Qux += np.dot(self.nextVx[j].item(),self.model.fux[j])
                    self.Quu += np.dot(self.nextVx[j].item(),self.model.fuu[j])
                
                self.QuuInv = np.linalg.inv(self.Quu)

                self.k = - np.dot(self.QuuInv,self.Qu)
                self.K = - np.dot(self.QuuInv,self.Qux)

                # regularization (Y. Tassa thesis)
                self.nextVx      = self.Qx + np.dot(np.dot(self.K.T,self.Quu),self.k) + np.dot(self.K.T,self.Qu) + np.dot(self.Qux.T,self.k)
                self.nextVxx     = self.Qxx + np.dot(np.dot(self.K.T,self.Quu),self.K) + np.dot(self.K.T,self.Qux) + np.dot(self.Qux.T,self.K)
                
                if(np.all(np.linalg.eigvals(self.Quu) <= 0)): 
                    # Ending condition for backwardPass
                    # Checking whether Quu is positive definite or not
                    self.mu = self.mu*10
                    self.completeBackwardFlag = False
                    break
                else:
                    self.mu = self.mu/10
                    
                self.kList.append(self.k)
                self.KList.append(self.K)
                
            self.kList.reverse()
            self.KList.reverse()
            return 0
        
    def forwardLoop(self):
        self.nextXList = [self.Xinit]
        self.nextUList = []

        self.changeAmount = 0.0
        self.nextXList[0] = self.Xinit
        # Line search to be implemented
        self.alpha = self.alphaList[0]
        for i in range(self.T):
            self.nextUList.append(self.UList[i] + self.alpha*self.kList[i] + np.dot(self.KList[i],(self.nextXList[i] - self.XList[i])))
            self.model.computeNextState(self.dt,self.nextXList[i],self.nextUList[i])
            self.nextXList.append(self.model.nextX)
            for j in range(self.model.commandNumber):
                self.changeAmount += np.abs(self.UList[j] - self.nextUList[j])
        return 0
                    
