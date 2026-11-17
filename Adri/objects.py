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
    A/g : np.ndarray or FunctionType
        The matrix A for the linear ODE system 
        or a function g(t, y) for a generic ODE system.
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
    def __init__ (self, t0: float, y0: np.ndarray, T: float, g: FunctionType):
        super ().__init__(t0 = t0, T = T, y0 = y0)
        self.g = g

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Computes the ODE system function.
        """
        return self.g(t, y)




class LoktaVolterra(IVP):
    def __init__ (self, t0: float, y0: np.ndarray, T: float, a: float, b: float, c: float, d: float):
        super ().__init__(t0 = t0, T = T, y0 = y0)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Lokta-Volterra model function.
        """
        y0 = self.a*y[0] - self.b*y[0]*y[1]
        y1 = self.c*y[0]*y[1] - self.d*y[1]

        return np.array([y0,y1])




class vanderPol(IVP):
    def __init__ (self, t0: float, y0: np.ndarray, T: float, mu: float):
        super ().__init__(t0 = t0, T = T, y0 = y0)
        self.mu = mu

    def f(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        van der Pol equation. 
        """
        y0 = y[1]
        y1 = self.mu*(1-y[0]**2)*y[1] - y[0]

        return np.array([y0,y1])



    
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
    tol : float
        The error tolerance used by adaptive step-size control.
    """
    def __init__(self, ivp: IVP, N: int, 
                tol: float = 1e-6, adaptative: bool=False):
        self.ivp = ivp
        self.N = N
        self.grid = np.linspace(self.ivp.t0, self.ivp.T, N + 1)
        self.h = (self.ivp.T - self.ivp.t0) / N
        self.u = np.zeros((N + 1, len(self.ivp.y0)))
        self.u[0] = self.ivp.y0
        self.adapt = adaptative
        
        # Adaptative step initialize parameters
        if self.adapt:          
            self.tol = tol
            self.N = 1

            T = self.ivp.T
            t0 = self.ivp.t0
            y0 = self.u[0]

            self.grid = np.array([t0])
            self.u = np.array([y0])
            self.h = ( np.abs(T-t0)*(self.tol**(1/self.k)) )/ \
                ( 100 + 100*np.linalg.norm(self.ivp.f(t0,y0)) )

    def step(self, tn: float, un: np.ndarray) -> np.ndarray:
        """
        Solves one step of the discrete IVP. 
        The step method must be implemented by subclasses of Solver.
        """
        raise NotImplementedError("The step method must be implemented")

    def integrate(self):
        """
        Integrate the discrete IVP
        """
        if not self.adapt:
            for i in range(self.N):
                self.u[i + 1] = self.step(tn = self.grid[i], un = self.u[i])
        else:
            stop = False
            while not stop:
                self.N += 1
                new_grid_val = self.h+self.grid[-1]
                if new_grid_val > self.ivp.T:
                    new_grid_val = self.ivp.T
                    self.h = self.ivp.T-self.grid[-1]
                    stop = True

                new_u_val = self.step(tn = self.grid[-1], un = self.u[-1])
                self.u = np.vstack((self.u, new_u_val))
                self.grid = np.append(self.grid,new_grid_val)

                self.newstep()



    def newstep(self):
        """Computes the next step size from the current and previous error estimates."""
        h_new = (
                (self.tol / self.errest) ** (2 / (3 * self.k))
                * (self.tol / self.errestold) ** (-1 / (3 * self.k))
                * self.h
            )
        self.errestold = self.errest
        self.h = h_new
        if not np.isfinite(h_new) or h_new <= 0:
            raise RuntimeError(f"Step size update must be positive or finite."+
                               f"\nUpdate:\n{h_new}")




class ExplicitEuler(Solver):
    def __init__ (self, ivp: IVP, N: int):
        super ().__init__(ivp = ivp, N = N)
        
    def step(self, tn: float, un: np.ndarray) -> np.ndarray:
        return un + self.h * self.ivp.f(t = tn, y = un)




class ImplicitEuler(Solver):
    def __init__ (self, ivp: IVP, N: int):
        super ().__init__(ivp = ivp, N = N)
        
    def step(self, tn: float, un: np.ndarray) -> np.ndarray:
        A = self.ivp.A 
        return scipy.linalg.solve(np.eye(A.shape[0]) - self.h * A, un)




class Trapezoidal(Solver):
    def __init__ (self, ivp: IVP, N: int):
        super ().__init__(ivp = ivp, N = N)
        
    def step(self, tn: float, un: np.ndarray) -> np.ndarray:
        A = self.ivp.A 
        return scipy.linalg.solve(np.eye(A.shape[0]) - 0.5 * self.h * A, un + 0.5 * self.h * (A @ un))




class RK4(Solver):
    def __init__ (self, ivp: IVP, N: int):
        super ().__init__(ivp = ivp, N = N)
        
    def step(self, tn: float, un: np.ndarray) -> np.ndarray:
        f = self.ivp.f
        h = self.h
        y1 = f(tn, un)
        y2 = f(tn + h/2, un + h/2 * y1)
        y3 = f(tn + h/2, un + h/2 * y2)
        y4 = f(tn + h, un + h * y3)

        return un + h/6 * (y1 + 2*y2 + 2*y3 + y4)




class RK34(Solver):
    def __init__ (self, ivp: IVP, N: int, tol: float=1e-6, adaptative: bool=False):
        self.errest = None    # Initialize the error estimate attribute
        self.errestold = tol  # Initialize the previous error estimate attribute
        self.k = 4            # Order of local truncation error
        super ().__init__(ivp = ivp, N = N, tol = tol, adaptative= adaptative)
        
        
    def step(self, tn: float, un: np.ndarray) -> np.ndarray:
        f = self.ivp.f
        h = self.h
        y1 = f(tn, un)
        y2 = f(tn + h/2, un + h/2 * y1)
        y3 = f(tn + h/2, un + h/2 * y2)
        z3 = f(tn + h, un - h * y1 + 2 * h * y2)  
        y4 = f(tn + h, un + h * y3)

        local_error_est = h/6 * (2*y2 + z3 - 2*y3 - y4)
        self.errest = np.linalg.norm(local_error_est)

        return un + h/6 * (y1 + 2*y2 + 2*y3 + y4)





class Evaluator():
    """
    Class for evaluating errors of a numerical method on an IVP.
    Class should be called with an initialized IVP class,
    and an unitialized Solver class.
    """
    def __init__(self, ivp: IVP, solver: Solver):
        self.ivp = ivp
        self.solver = solver

    def errvstime(self,N: int,relative: bool=False) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the error between the numerical solution and the exact solution at each time step.
        """
        if not hasattr(self.ivp, 'exact'):
            raise NotImplementedError("The exact solution method must be implemented in the IVP class.")
        
        solver = self.solver(ivp = self.ivp, N = N)
        solver.integrate()
        
        exact_solution = np.array([self.ivp.exact(t) for t in solver.grid])
        error = np.linalg.norm(solver.u - exact_solution, axis=1)

        if relative:
            error /= np.linalg.norm(exact_solution, axis=1)

        return solver.grid, error

    def errvsh(self, N: np.array, relative: bool=False) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the error between the numerical solution and the exact solution 
        in the final step for different fixed step sizes.
        """
        if not hasattr(self.ivp, 'exact'):
            raise NotImplementedError("The exact solution method must be implemented in the IVP class.")
        
        errors = []
        hs = []
        
        for n in N:
            solver = self.solver(ivp = self.ivp, N = n)
            solver.integrate()
            
            exact_solution = self.ivp.exact(solver.grid[-1])
            error = np.linalg.norm(solver.u[-1] - exact_solution)  # Error at final time
            
            if relative:
                error /= np.linalg.norm(exact_solution)
            
            errors.append(error)
            hs.append(solver.h)
        
        return np.array(hs), np.array(errors)