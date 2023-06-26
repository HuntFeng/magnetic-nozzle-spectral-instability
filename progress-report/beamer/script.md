# Title page
Hi Professor Andrei, Chijin and Michael, my name is Hunt Feng. Thank you for taking time to listen to my presentation. In this presentation I am going to talk about the work I have done for investigating the instability of the plasma flow in magnetic nozzle.

# Introduction
## Instability of Plasma Flow
Before we dive into the work I work, I will first explain some words on the title.  

The first concept is the instability of plasma flow. To put it short, instability of plasma flow means the tendency of a plasma system to deviate from equilibrium state given small perturbations.

To make a machanical analogy, imagine a ball on a hill. If we give a small push, the ball will fall downhill. Therefore the system is unstable.

A common way to investigate instability of a plasma system, is to assume the perturbations are small and oscillatory. Then we solve for frequency $\omega$. By examining the imaginary part of $\omega$, so-called the growth rate, we can determine the instability of a plasma system.

If the growth rate is larger than 0, then the perturbations will grow exponential in time. Hence the system is unstable. Otherwise, the perturbations decays or stays the same over time. Hence the system is stable.

## Magnetic Nozzle
The next concept I will introduce is magnetic nozzle. It is a propulsion device that uses magnetic field to control the plasma flow. We can see, the plasma flow enters the nozzle from the left, and it will be acceleraeted by the magnetic field, and exhaust on the right. This magnetic field here acts like the wall of a Laval nozzle, hence the name magnetic nozzle.

The instability of plasma flow in magnetic nozzle may affect the operation of a magnetic nozzle. Hence it is worth to investigate the instability of different plasma flows under different boundary conditions.  

## Governing Equations
To investigate the instability of plasma flows in magnetic nozzle. We need to first find out the equilibrium flows. We can do that by modeling the plasma flow using conservation of density and momentum to in the framework of fluid, where $n$ and $v$ are density and velocity, respectively.

Since the equilibrium flow is a time independent solution to this system. So we know that the density and velocity of an equilibrium flow must satify the so-called equilibrium condition. 

## Equilibrium Velocity Profiles
By solving the equilibrium condition, we get the velocity profile, $v_0$. There are 4 different cases. Subsonic case, the plasma flow enters and exits the nozzle with subsonic velocity. Supersonic case similar to that but with supersonic velcity. Accelerating case, the plasma flow enters the nozzle with subsonic speed and accelerates to supersonic speed on the exit. Decelerating case is similar but opposite.    

## Polynomial Eigenvalue Problem
As we mentioned before, a common trick to investigate instability is to set the perturbations oscillatory. By substituting them into the linearized governing equations. We are able to transform an instability problem to a so-called polynomial eigenvalue problem. 

Given the velocity profile of an equilibrium plasma flow, $v_0$, we need to solve the small oscillatory perturbation, $\tilde{v}$ to that and its oscillating frequency $\omega$. In the language of mathematics, $\omega$ is an eigenvalue of these operators, and $\tilde{v}$ is an eigenfunction associated with eigenvalue $\omega$. 

My main task in this research is to solve this polynomial eigenvalue problem.

# Spectral Method
## Spectral Method
The problem seems to be easy at first glance. We decided to use spectral method as our approach. Since it is often a good approach when it comes to eigenvalue problems. 

Our first step is to reformulate the polynomial eigenvalue problem, where the operators $\hat{M}$ and $\hat{N}$ are these two huge truncks. We can see it takes the form of a standard eigenvalue problem now. An operator acts on an unknown function, and spits out a scaled unknown function.

By discretizing the operators using finite-difference, finite-element, or spectral-element, this problem becomes a simple algebraic eigenvalue problem, a matrix multiplys an eigenvector and spits out a scaled eigenvector. We can then solve it using numerical linear algebra package like `linalg` submodule in `numpy`.  

## Spectral Pollution
However, we quickly ran into trouble even for the simpliest setup.

If we set the equilibrium velocity profile $v_0$ as constant. Then we can actually solve the problem analytically and all modes are stable. To be more specific, the eigenvalues are all real, meaning that the growth rates are zero.

Yet the numerical results show spurious unstable modes. These unstable modes are fake, and they occur due to the inappropriate discretization of operators. This phenomenon is so-called spectral pollution, and it occurs regardless of the resolution of discretization. Moreover, I observe this phenomenon in all the discretization schemes I chose. On the figure, I only showed the results obtained by utilizing finite-difference.

## Filtering Spurious Modes
Fortunatly, we can filter the spurious modes by doing convergence tests. The idea is simple, we solve for the eigenvalues using the same discretization under different resolution. We can easily spot the convergent eigenvalues if we plot eigenvalues of different resolutions on the same graph. By doing this we are able to get the correct numerical results for the constant equilibrium velocity case. 

It is worth to mention, spectral pollution happens in most of the cases. So the convergence test becomes a must-do after each solve.

# Singularity Perturbation
## Existence of Singularity
Spectral method together with convergence test is good for most of the cases. However, when dealing transonic velocity profile, it does not work well. Take accelerating case for example. We can see the eigenfunctions are squeezed to the center, which is the neck of the magnetic nozzle. It is hard to tell if the solution is correct or not.

Later on we noticed the existence of singularity at the neck of the nozzle. Since the accelerating velocity profile crosses the sonic point, so the second order derivative term in the polynomial eigenvalue problem vanishes. $z=0$ is in fact a regular singular point after we performed some standard ODE analysis.

## Interesting Connection to Black Hole
Here we must mention a very interesting point. The sonic horizon, which forms when fluid exceeds the sound speed, is an exact analogue of a black hole horizon. In the papers discussing the connection between black hole and de Lavel nozzle, they use the acoustic analogue of tortoise coordinate so that their problem can be reformulated as a Schr$\text{\"o}$dinger-type equation. 

## Shooting method
The acoustic analogue of tortoise coordinate is too complicated to use. We can use a simpler approach, shooting method, since the singularity is a regular singularity.

We use the Frobenius method instead as our approach. We assume the eigenfunction to be a power series, and then expand it at the singularity. We are able to pick up the regular solution and setup the boundary condition for the eigenfunction at the singularity.

In the accelerating case, the next step is to shoot the eigenfunction to the left and we can find the eigenvalues by matching the eigenfunctions to the boundary condition on the left. In this figure, I set Dirichlet boundary condition on the left. We see that all modes cross the singularity smoothly, and they are all stable.


# Future Work
## Future Work
This research can go several directions later. One is to setup non-zero boundary condition on the left for the accelerating velocity profile, and interpret the physical meaning of the eigenvalues and eigenfunctions.

On the other hand, we will compare the results to analytically solvable problems with similar configurations and see if we can get any insights.

# Ending
My presentation ends here. Thank you all again. Any questions?
