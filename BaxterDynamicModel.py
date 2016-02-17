import numpy as np
from random import uniform
import time

class BaxterDynamicModel:
    """
        dynamic model class for moving baxter arms
        - input of iLQRSolver
        - return derivatives
        - return nextState from currentState and chosen Action
        states - 3D coordinates of end effector of baxter
        actions - 6 directions(forward, backward, left, right, up, down)
        
    """
    
    def __init__(self):
        self.X = np.zeros((3,1))
        self.U = np.zeros((3,1))
        self.nextX = np.zeros((3,1))
        
        # 3-Dim Coordinates
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        
        self.stateNumber = 3
        self.commandNumber = 3
        
        self.fx = np.zeros((3,3))
        self.fu = np.zeros((3,3))
        
	self.B = np.matrix([[1.0, 0.0, 0.0],
			    [0.0, 1.0, 0.0],
			    [0.0, 0.0, 1.0]
			])
	self.Bd = np.matrix([[1.0, 0.0, 0.0],
			    [0.0, 1.0, 0.0],
			    [0.0, 0.0, 1.0]
			])
               
        # fxx 3x3  x 3
        self.fxx = list()
        self.fxx.append(np.zeros((3,3)))
        self.fxx.append(np.zeros((3,3)))
        self.fxx.append(np.zeros((3,3)))
        
        # fuu 3x1  x 3
        self.fuu = list()
        self.fuu.append(np.zeros((3,3)))
        self.fuu.append(np.zeros((3,3)))
        self.fuu.append(np.zeros((3,3)))
                        
        self.fux = list()
        self.fux.append(np.zeros((3,3)))
        self.fux.append(np.zeros((3,3)))
        self.fux.append(np.zeros((3,3)))
        
        self.fxu = list()
        self.fxu.append(np.zeros((3,3)))
    
    def computeNextState(self,dt,X,U):
        self.Bd = dt*self.B
	#print self.Bd
	#print "-----"
	#print U
        self.nextX = np.dot((self.Bd + np.eye(3)),self.X) + np.dot(self.Bd,U)
        #self.nextX = X + U
        return 0
    
    def computeAllModelDeriv(self, dt, X, U):
        self.fx = self.Bd + np.eye(3)
        self.fu = self.B*dt
        return self.fx, self.fxx, self.fu, self.fuu, self.fxu, self.fux
