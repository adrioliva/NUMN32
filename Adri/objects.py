import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
from types import FunctionType




class IVP:
    """
    Class representing an Initial Value Problem (IVP) 
    for an Ordinary Differential Equation (ODE) system.

    ------
    INPUTS
    ------
    t0 : float
        The initial time.
    y0 : np.ndarray
        The initial state vector.
    T : float
        The final time for the simulation.
    """
    def __init__(self, t0: float, y0: np.ndarray, T: float):
        self.t0 = t0
        self.y0 = y0
        self.T = T

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Function that defines the ODE system.
        """
        raise NotImplementedError("The function f must be implemented in the inherited class.")




class LTE(IVP):
    def __init__ (self, t0: float, y0: np.ndarray, T: float, A: np.ndarray):
        super ().__init__(t0 = t0, T = T, y0 = y0)
        self.A = A

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Computes the test linear ODE system function.
        """
        return self.A @ y

    def exact(self, t: float) -> np.ndarray:
        """
        Computes the exact solution of the test linear ODE system.
        """
        return sp.linalg.expm(self.A * (t - self.t0)) @ self.y0




class Generic(IVP):
    def __init__ (self, t0: float, y0: np.ndarray, T: float, f: FunctionType):
        super ().__init__(t0 = t0, T = T, y0 = y0)
        self.g = f

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Computes the generic linear ODE system function.
        """
        return self.g(t, y)




class Solver:
    """
    Class representing a numerical solver for an IVP.
    ------
    INPUTS
    ------
    ivp : IVP
        An instance of the IVP class representing the initial value problem.
    N : int
        The number of time steps.
    """
    def __init__(self, ivp: IVP, N: int):
        self.ivp = ivp
        self.N = N
        self.grid = np.linspace(self.ivp.t0, self.ivp.T, N + 1)
        self.h = (self.ivp.T - self.ivp.t0) / N
        self.u = np.zeros((N + 1, len(self.ivp.y0)))
        self.u[0] = self.ivp.y0

    def step(self, tn: float, un: np.ndarray) -> np.ndarray:
        """
        Solves one step of the discrete IVP. 
        The step method must be implemented by subclasses of Solver.
        """
        raise NotImplementedError("The step method must be implemented")

    def integrate(self):
        for i in range(self.N):
            self.u[i + 1] = self.step(tn = self.grid[i], un = self.u[i])




class ExplicitEuler(Solver):
    def __init__ (self, ivp: IVP, N: int):
        super ().__init__(ivp = ivp, N = N)
        
    def step(self, tn: float, un: np.ndarray) -> np.ndarray:
        return un + self.h * self.ivp.f(t = tn, y = un)

    