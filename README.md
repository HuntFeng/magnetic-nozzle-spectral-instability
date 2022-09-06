# Instability In Magnetic Nozzle
## General methods
| Method(file)\Applicable to | Constant $v$ Problem | Original Problem | Result |
|-|-|-|-|
| Finite Difference <br> `finite-difference.ipynb` | Yes | Yes | Subsonic: stable <br> Supersonic: spuriously unstable <br> Transonic: spuriously unstable |
| Finite Element <br> `finite-element.ipynb` | Yes | Yes | Subsonic: stable <br> Supersonic: spuriously unstable <br> Transonic: spuriously unstable |
| DVR <br> `dvr-method.ipynb` | Yes | Yes | Subsonic: stable <br> Supersonic: spuriously unstable <br> Transonic: spuriously unstable |
| Reduced to Normal Form <br> `normal-form.ipynb` | Yes | No | Subsonic: stable <br> Supersonic: stable |
| Analytical Solution <br> `exact-solution.ipynb` | Yes | No | Subsonic: stable <br> Supersonic: stable |

## Filtering techniques
| Technique\Applicable to | Constant $v$ Problem | Original Problem | Result |
|-|-|-|-|
| Add Diffusion <br> `finite-difference.ipynb` | Yes | Yes | Filtered most spurious modes except low $k$ unstable modes |
| Spectral Pollution Theory <br> `convergence-test.ipynb` | Yes | No | Eliminates all spurious modes |
| Filter By Convergence <br> `convergence-test.ipynb` | Yes | Yes | Eliminates all spurious modes |
