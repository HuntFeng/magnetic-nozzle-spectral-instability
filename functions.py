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
        D1 = np.diag(0.5/h*np.ones(N-1), k=1) + np.diag(-0.5/h*np.ones(N-1), k=-1)
        D2 = np.diag(-2/h**2*np.ones(N), k=0) + np.diag(1/h**2*np.ones(N-1), k=1) + np.diag(1/h**2*np.ones(N-1), k=-1)
        
        # 6th-order finite difference
        # coeff_D1 = [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]
        # coeff_D2 = [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
        # D1 = np.zeros((N,N))
        # D2 = np.zeros((N,N))
        # ones = np.ones(N)
        # for k in range(-3,4):
        #     deviate = np.abs(k)
        #     ind = k+3
        #     D1 += np.diag(coeff_D1[ind]*ones[:N-deviate], k=k)/h
        #     D2 += np.diag(coeff_D2[ind]*ones[:N-deviate], k=k)/h**2
        
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


def compute(plasma:Plasma, spectral:Spectral, ky_range: ArrayLike):
    # differentiation matrices
    N = spectral.N
    x = spectral.x
    D1 = spectral.D1[1:,1:]
    D2 = spectral.D2[1:,1:]
    I = np.eye(*D1.shape)

    # model parameters
    #ky_range = np.linspace(1,151, 51) # unit: m^{-1}, wave number in y direction
    L = 0.1 # unit: m, system length

    # results arrays
    gamma = np.nan*np.empty((int(np.ceil(N/2)+1), ky_range.size)) # corresponding growth rates, Im(\omega)
    omega = np.nan*np.empty((int(np.ceil(N/2)+1), ky_range.size)) # corresponding frequency, Re(\omega)
    #nodes = np.nan*np.empty((int(np.ceil(N/2)+1), ky_range.size)) #  number of nodeds for each eigenfunction corresponding to given ky

    gamma_max = np.nan*np.empty((ky_range.size,)) # max Im(\omega) at each ky
    omega_max = np.nan*np.empty((ky_range.size,)) # max Re(\omega) at each ky
    eigvec = np.nan*np.empty((D1.shape[0], ky_range.size), dtype=complex) # phi

    for i, ky in enumerate(tqdm(ky_range)):
        # normalized quantities
        v0 = plasma.v0/plasma.cs

        # matrices for polynomial eigenvalue problem
        A2 = I
        A1 = 2*(np.diag(v0)@D1 + np.diag(D1@v0))
        A0 = np.diag(1-v0**2)@D2 \
            - np.diag((3*v0 + 1/v0)*(D1@v0))@D1 \
            - np.diag((1-1/v0**2)*(D1@v0)**2) \
            - np.diag((v0+1/v0)*(D2@v0))

        V, e = polyeig(A0, A1, A2)
        
        # select only positive imaginary part
        ind = np.imag(e)>5/plasma.w_LH
        V = V[:, ind]
        e = e[ind]

        if e[np.abs(np.imag(e))>0].size > 0:
            ind = np.argmax(np.imag(e))
            gamma_max[i] = np.imag(e[ind])
            omega_max[i] = np.real(e[ind])
            eigvec[:,i] = V[:,ind]

            nodes_temp = np.zeros(e.size, dtype=int)
            for j in range(e.size):
                v = np.pad(np.real(V[:,j]), 1, constant_values=[0])
                peaks, _ = find_peaks(np.abs(v))
                nodes_temp[j] = peaks.size + 1
            
            # sort by number of nodes in descending order
            ind = nodes_temp.argsort()[::-1]
            nodes_temp = nodes_temp[ind]
            gamma[nodes_temp, i] = np.imag(e[ind])
            omega[nodes_temp, i] = np.real(e[ind])
            #nodes[nodes_temp, i] = nodes_temp
        else:
            eigvec[:,i] = 0

    # group the results for convenience
    result = {
        "spectral": spectral,
        "ky_range": ky_range,
        "gamma":gamma,
        "omega":omega,
        #"nodes": nodes,
        "gamma_max": gamma_max,
        "omega_max": omega_max,
        "eigvec": eigvec}
    return result


def linear_fill(a):
    """
    Fill NaNs between first and last finite values column by column
    """
    for j in range(a.shape[1]):
        finite = np.argwhere(np.isfinite(a[:,j])).flatten()
        if finite.size != 0: # if there are finite values
            # between the first and last finite values in the row, we linearly fill the nan
            interp_vals = np.interp(np.arange(finite[0],finite[-1]), finite, a[finite,j])
            a[finite[0]:finite[-1],j] = interp_vals

def plotting(result):
    spectral = result["spectral"]
    x = spectral.x
    N = spectral.N
    ky_range = result["ky_range"]
    gamma = result["gamma"]
    omega = result["omega"]
    #nodes = result["nodes"]
    gamma_max = result["gamma_max"]
    omega_max = result["omega_max"]
    eigvec = result["eigvec"]

    # omega-ky
    plt.figure()
    plt.plot(ky_range, omega_max, label="$\\Re(\\omega)$")
    plt.plot(ky_range, gamma_max, label="$\\Im(\\omega)$")
    plt.xlabel("$k_y, m^{-1}$")
    plt.ylabel("$\\Re(\\omega), \\Im(\\omega)$")
    plt.legend();

    # phi-x
    plt.figure()
    for i in range(4):
        real_part = np.pad(np.real(eigvec[:,i]), 1, constant_values=[0])
        imag_part = np.pad(np.imag(eigvec[:,i]), 1, constant_values=[0])
        lines = plt.plot(x, real_part, '-', label=f"$k_y={ky_range[i]}$")
        plt.plot(x, imag_part, '--', color=lines[0]._color)

    plt.xlabel("$x$")
    plt.ylabel("$\\phi$")
    plt.legend();

    # phase-space
    plt.figure()
    for i in range(4):
        plt.plot(omega[:,i], gamma[:,i], 'o', label=f"$k_y={ky_range[i]}$")

    plt.xlabel("$\\Re(\\omega)$")
    plt.ylabel("$\\Im(\\omega)$")
    plt.legend();

    # kx-ky
    plt.figure()
    gamma_ = gamma.copy()
    linear_fill(gamma_) # fill the missing values with linear interpolation
    plt.pcolormesh(gamma_, cmap='jet')
    # change the tick density to 1/10 and the tick labels to their corresponding values
    plt.yticks(range(1,gamma_.shape[0]+1,10), range(1,N+1,20))
    plt.xticks(range(1,gamma_.shape[1]+1,10), ky_range[0::10].astype(int))
    plt.colorbar(label="$\\Im(\\omega)$")
    plt.xlabel("$k_y$")
    plt.ylabel("$Nodes, k_x$");