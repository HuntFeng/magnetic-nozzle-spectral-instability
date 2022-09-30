import numpy as np
import scipy.linalg as sl
from scipy.special import lambertw
from dataclasses import dataclass
import matplotlib.pyplot as plt

# set mpl settings at runtime
import json

with open("./mpl_config.json") as fp:
    config = json.load(fp)
    for k,v in config.items():
        plt.rcParams[k] = v

class Spectral:
    def __init__(self, N:int, domain:str, method:str) -> None:
        """
        domain: either "symmetric" or "nonsymmetric"
        method: either "FD" (finite difference) or "CH" (chebyshev)
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


@dataclass
class Params:
    """
    Problem config
    constant_v: is this problem constant v ?
    accelerating: is this accelerating case ?
    Mm: mid velocity

    Magnetic field 
    B0: amplitude of B
    R: mirror ratio
    Bm: mid point B
    Delta: width of B peak
    """
    Mm: float
    constant_v: bool = None
    accelerating: bool = None

    B0: float = 1
    R: float = 1.5
    Bm: float = 1+R
    Delta: float = 0.1/0.3

    def __post_init__(self):
        if (self.constant_v is None) and (self.accelerating is None):
            raise RuntimeError("Must initialize at least one of `constant_v` and `accelerating` ")
        

class Nozzle:
    def __init__(self, params: Params, x: np.array, u: callable=None) -> None:
        """ 
        Investigate the instability of magnetic nozzle
        
        Input:
            params: physical parameters and experiment setup
            x: mesh of the magnetic nozzle
            u(x,n): if None, we are using finite difference; if provided, we are using finite element
        """
        
        self.params = params # params
        self.x = x # mesh
        self.u = u # trial functions for finite element
        self.v0 = self.velocity_profile(x)

    def velocity_profile(self, x):
        """ 
        For convience of DVR method, make v0 a function of x 
        input:
            x
        output:
            v0
        """
        Mm = self.params.Mm
        constant_v = self.params.constant_v
        accelerating = self.params.accelerating

        B0 = self.params.B0
        R = self.params.R
        Bm = self.params.Bm
        Delta = self.params.Delta

        # velocity normalized to sound speed
        # k=-1: supersonic branch
        # k=0: subsonic branch
        B = lambda x: B0*(1+R*np.exp(-(x/Delta)**2))
        W = lambda x,k: np.real(lambertw(x,k=k)) # I only need the real parts
        M = lambda x, Mm, k: np.sqrt( -W(-Mm**2 * (B(x)/Bm)**2 * np.exp(-Mm**2), k=k) )

        if constant_v:
            v0 = Mm*np.ones_like(x) # constant v=0.1
        else:
            if Mm < 1:
                v0 = M(x, Mm=Mm, k=0) # subsonic velocity profile, M_m < 1
            elif Mm == 1:
                # transonic profile, accelerating/decelerating
                mid_point = [] if (x.size % 2 ==0) else [1]
                if accelerating:
                    v0 = np.concatenate([M(x[x<0], Mm=1, k=0), mid_point, M(x[x>0], Mm=1, k=-1)]) # accelerating velocity profile
                else:
                    v0 = np.concatenate([M(x[x<0], Mm=1, k=-1), mid_point, M(x[x>0], Mm=1, k=0)]) # decelerating velocity profile
            else:
                v0 = M(x, Mm=Mm, k=-1) # supersonic velocity profile, M_m > 1
        return v0

    def polyeig(self, *A: np.array):
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

    def solve(self, *matrices: np.array):
        """ 
        If len(matrices)==1, then we are solving A@v = lambda*v 
        If len(matrices)>1, then we are solving polynomial eigenvalue problem
        If self.u==None, using finite difference discretization
        If self.u!=None, using finite element discretization
        """
        if not self.u:
            # finite difference
            if len(matrices) == 1:
                self.omega, V = np.linalg.eig(matrices[0])
                # only need half of the eigenvector
                V = V[:int(V.shape[0]/2)]
            else:
                V, self.omega = self.polyeig(*matrices)
            # Dirichlet boundary condition
            self.V = np.pad(V, ((1,1),(0,0)))
        else:
            # finite element
            if len(matrices) == 1:
                self.omega, C = np.linalg.eig(matrices[0])
                # only need half of the eigenvector
                C = C[:int(C.shape[0]/2)]
            else:
                C, self.omega = self.polyeig(*matrices)
            # Dirichlet boundary condition
            C[0,:] = 0
            C[-1,:] = 0
            self.V = np.zeros((self.x.size, C.shape[1]), dtype=complex)
            for i in range(C.shape[1]):
                for n in range(C.shape[0]):
                    self.V[:,i] += C[n,i]*self.u(self.x, n)
    
    def sort_solutions(self, real_range: list=[0,50], imag_range: list=[]):
        selection = (self.omega.real > real_range[0]) & (self.omega.real < real_range[1])
        self.omega = self.omega[selection]
        self.V = self.V[:,selection]
        if imag_range:
            selection = (self.omega.imag > imag_range[0]) & (self.omega.imag < imag_range[1])
            self.omega = self.omega[selection]
            self.V = self.V[:,selection]
        
        ind = np.argsort(self.omega.real)
        self.omega = self.omega[ind]
        self.V = self.V[:,ind]

    def plot_eigenvalues(self, ax=None):
        if not ax:
            _, ax = plt.subplots()
        ax.plot(self.omega.real, self.omega.imag, 'o')
        ax.set_xlabel("$\Re(\omega)$")
        ax.set_ylabel("$\Im(\omega)$")
        return ax

    def plot_eigenfunctions(self, num_funcs:int=3, ax=None):
        if not ax:
            _, ax = plt.subplots()
        for i in range(num_funcs):
            line = ax.plot(self.x, self.V[:,i].real/np.abs(self.V[:,i].real).max(), label=f"$\omega=${self.omega[i]:.3f}")
            ax.plot(self.x, self.V[:,i].imag/np.abs(self.V[:,i].imag).max(), '--', color=line[-1].get_color())
            ax.set_xlabel("$z$")
            ax.set_ylabel("$\\tilde{v}$")
        ax.legend()
        return ax


if __name__ == '__main__':
    N = 101
    spectral = Spectral(N, "symmetric", "FD")
    params = Params(Mm=0.5, constant_v=True)
    nozzle = Nozzle(params, spectral.x)
    v0 = nozzle.v0
    x = spectral.x
    D1 = spectral.D1
    D2 = spectral.D2

    I = np.eye(*D1.shape)
    A11 = np.zeros_like(D1)
    A12 = I
    A21 = -np.diag(1-v0**2)@D2 \
            + np.diag((3*v0 + 1/v0)*(D1@v0))@D1 \
            + np.diag((1-1/v0**2)*(D1@v0)**2) \
            + np.diag((v0+1/v0)*(D2@v0))
    A22 = -2j*(np.diag(v0)@D1 + np.diag(D1@v0)) #- eta*np.diag(v0)@D2

    A = np.block([[A11[1:-1,1:-1], A12[1:-1,1:-1]],[A21[1:-1,1:-1], A22[1:-1,1:-1]]])
    nozzle.solve(A)
    nozzle.sort_solutions()
    nozzle.plot_eigenvalues()
    plt.show()