from turtle import right
import constants as const
import numpy as np
from numpy.typing import ArrayLike
import scipy.linalg as sl
from scipy.signal import find_peaks
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10,10)
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 16
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

class Spectral:
    def __init__(self, N:int, domain:str, method:str) -> None:
        """
        domain: either "symmetric" or "nonsymmetric"
        method: either "FD" (finite difference) or "CH" (chebyshev)
        boundary_condition: {"left":left_type, "right":right_type} 
        where type can be either Dirichlet or Neumann
        """
        self.N = N
        self.domain = domain
        if method == "FD":
            self.x, self.D1, self.D2 = self.matrix_FD()
        if method == "CH":
            self.x, self.D1, self.D2 = self.matrix_CH()


    def matrix_FD(self):
        """
        Finite difference differentiation matrix d/dx
        Central difference method.
        
        input: 
            N(int) number of grids
            domain(str): can be "symmetric" or "nonsymmetric"
        output:
            grid points array x
            differential operators d/dx and d^2/dx^2 in matrix form 
        """
        N = self.N 
        domain = self.domain
        if domain == "symmetric":
            x = np.linspace(-1,1,N)
            h = np.abs(x[1]-x[0])
        elif domain == "nonsymmetric":
            h = 1/N
            x = np.arange(h,1+h,h)
        else:
            raise NameError(f"Valid domain types are 'symmetric' and 'nonsymmetric'")

        # 2nd-order finite difference
        # D1 = np.diag(0.5/h*np.ones(N-1), k=1) + np.diag(-0.5/h*np.ones(N-1), k=-1)
        # D2 = np.diag(-2/h**2*np.ones(N), k=0) + np.diag(1/h**2*np.ones(N-1), k=1) + np.diag(1/h**2*np.ones(N-1), k=-1)
        
        # 6th-order finite difference
        coeff_D1 = [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]
        coeff_D2 = [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
        D1 = np.zeros((N,N))
        D2 = np.zeros((N,N))
        ones = np.ones(N)
        for k in range(-3,4):
            deviate = np.abs(k)
            ind = k+3
            D1 += np.diag(coeff_D1[ind]*ones[:N-deviate], k=k)/h
            D2 += np.diag(coeff_D2[ind]*ones[:N-deviate], k=k)/h**2
        
        return x, D1, D2


    def matrix_CH(self):
        """
        Chebyshev differentiation matrix d/dx
        
        input: 
            N(int) number of grids
            domain(str): can be "symmetric" or "nonsymmetric"
        output:
            grid points array x
            differential operators d/dx and d^2/dx^2 in matrix form 
        """
        N = self.N 
        domain = self.domain
        if domain == "symmetric":
            x = np.cos(np.pi*np.arange(N+1)/N)
        elif domain == "nonsymmetric":
            x = np.cos(np.pi*np.arange(N+1)/N)
            x = (x+1)/2
        else:
            raise NameError(f"Valid domain types are 'symmetric' and 'nonsymmetric'")
        x = x.reshape(-1,1) # to column vector

        c = np.pad(np.ones(N-1), 1, constant_values=[2]) * (-1)**np.arange(N+1)
        c = c.reshape(-1,1)
        X = np.repeat(x, N+1, axis=1) # make N+1 duplicated columns
        dX = X - X.T
        D = (c@(1/c).T) / (dX+np.eye(N+1)) # off-diagonal entries
        D = D - np.diag(np.sum(D.T, axis=0))
        D2 = D@D
        
        return x.flatten(), D, D2

class Plasma:
    def __init__(self, n:float,Te:float,Ti:float,Ln:float,Lb:float,E,B,fluid:str):
        """
        E and B fields can be float or function profile
        """
        # parameters
        self.n = n
        self.Te = Te
        self.Ti = Ti
        self.Ln = Ln
        self.Lb = Lb
        self.E = E
        B = B*1e-4 # convert Gauss to Tesla
        self.B = B
        self.fluid = fluid
        if fluid == "Ar":
            mi = 6.6335e-26
        elif fluid == "Xe":
            mi = 2.1801e-25
        self.mi = mi
        
        # velocities, m/s
        # self.v # the velocity profile is defined by user
        self.v0 = np.abs(E/B) # electric drift velocity
        self.cs = np.sqrt(const.kb*Te/mi) # ion sound velocity
        self.vTe = np.sqrt(const.kb*Te/const.me); # electron thermal velocity
        self.vTem = np.sqrt(8*const.kb*Te/(np.pi*const.me)); # mean electron thermal velocity
        self.vTi= np.sqrt(const.kb*Ti/mi); # ion thermal velocity
        self.vD= -2*((const.kb*Te)/(const.q*B))*Lb; # magnetic drift velocity
        self.vs= -((const.kb*Te)/(const.q*B))*Ln; # electron drift velocity
        self.vA = (B**2)/(const.mu0*n*mi); # Alfven velocity
        # frequencies, 1/s
        self.w_ce= (const.q*B)/(const.me);# electron gyrofrequency
        self.w_ci= (const.q*B)/(mi);# ion cyclotron
        self.w_pe = np.sqrt((const.q**2)*n/(const.me*const.eps0)); #plasma frequency
        self.w_pi = np.sqrt((const.q**2)*n/(mi*const.eps0)); #ion plasma frequency
        self.w_LH = (self.w_ce*self.w_pi)/np.sqrt(self.w_ce**2 +self.w_pe**2); # Low-hybrid frequency
        # lengthes, m
        self.re = self.vTe/self.w_ce; # electron gyroradius
        self.ri = self.vTi/self.w_ci; # ion gyroradius
        self.rs = self.cs/self.w_ci; # ion-sound Larmor radius   


def polyeig(*A):
    """
    Solve the polynomial eigenvalue problem:
        (e^0 A0 + e^1 A1 +...+  e^p Ap)x=0 

    Return the eigenvectors [x_i] and eigenvalues [e_i] that are solutions.

    Usage:
        X,e = polyeig(A0,A1,..,Ap)

    Most common usage, to solve a second order system: (K + C e + M e**2) x =0
        X,e = polyeig(K,C,M)

    """
    n = A[0].shape[0]
    l = len(A)-1 
    # Assemble matrices for generalized problem
    C = np.block([
        [np.zeros((n*(l-1),n)), np.eye(n*(l-1))],
        [-np.column_stack( A[0:-1])]
        ])
    D = np.block([
        [np.eye(n*(l-1)), np.zeros((n*(l-1), n))],
        [np.zeros((n, n*(l-1))), A[-1]]
        ])
    # Solve generalized eigenvalue problem
    e, X = sl.eig(C, D)
    if np.all(np.isreal(e)):
        e=np.real(e)
    X=X[:n,:]

    # Scaling each mode by max
    X /= np.tile(np.max(np.abs(X),axis=0), (n,1))
    return X, e 

def stability_condition(plasma, spectral, P,Q, omega_r, v_tilde):
    """
    Full instability condition from real part

    check stability condition for a specific omega and a specific v
    """
    mean = lambda y_vals: np.trapz(y_vals,spectral.x)/(spectral.x.max()-spectral.x.min())

    x, D1 = spectral.x, spectral.D1
    v0 = plasma.v

    v_sqr = np.real(v_tilde*v_tilde.conj())

    pdv_v = D1@v_tilde
    pdv_v[[0,-1]] = 0

    pdv_v0 = D1@v0
    pdv_v0[[0,-1]] = 0

    pdv_v_sqr = D1@v_sqr
    pdv_v_sqr[[0,-1]] = 0

    pdv_P = D1@P
    pdv_P[[0,-1]] = 0

    # coefficients of quadratic equation about gamma
    a = -mean(v_sqr)
    b = -mean(pdv_v0*v_sqr)
    c = omega_r**2*mean(v_sqr) \
        - 2*omega_r*mean(v0*np.imag(v_tilde.conj()*pdv_v)) \
        - mean((1-v0**2)*np.real(pdv_v.conj()*pdv_v)) \
        + mean((Q-0.5*pdv_P)*v_sqr)

    discriminant = b**2 - 4*a*c
    # if discriminant > 0:
    #     print(f"discriminant={discriminant:0.5f}>0, unstable")
    # elif discriminant < 0:
    #     print(f"discriminant={discriminant:0.5f}<0, stable")
    return discriminant


