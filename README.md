# Mixed Quantum-Classical Dynamics Yields Anharmonic Rabi Oscillations
Authors: Ming-Hsiu Hsieh, Roel Tempelaar

Publication: [*J. Chem. Phys.* **2025**, *162*, 224109](https://doi.org/10.1063/5.0266594)


## Usage of the code
The code has been run under Python 3.10.15 with the following packages of the version specified in parentheses.

Required packages: Numpy (1.26.4), Numba (0.60.0), Ray (2.38.0), matplotlib (3.9.2)

The Jupyter notebook file `diff_eq_plot.ipynb` includes the code for numerically solving the Duffing equation (Eq. 30 in the paper) using RK4, the following Fourier analysis, and plotting the figures in the paper.


To simulate MQC dynamics within the MF model, go to the `MQC` folder and execute:
```
python main.py
```
In the function `Wigner()` in `mixQC_MF_Foc.py`, you can choose to use Wigner sampling or the focused sampling.
The parameters can be changed in `input.py`.


Alternatively, to run a simulation with quantum exact reference (or CISD), go to `Quantum` folder and execute:
```
python Quantum_CISD.py
```
The parameters can be changed in `input.py`.
