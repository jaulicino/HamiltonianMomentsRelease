<h1> Hamiltonian Moments </h1>

The codes for https://arxiv.org/abs/2109.12790

<h2> Hamiltonian Class </h2>

The main functionality relies on the Hamiltonian object which the user can manipulate in a number of ways. Hamiltonians are defined as a linear combination of Pauli Strings and their respective coefficients. For instance,

```python
import Hamiltonians.hamiltonians as ham
Hamil = ham.Hamiltonian(["XZXZY, IIIIX"], [0.25,1.j])
```


The codes provide functionality for operator multiplication, multiplication by a scalar, addition of operators, and grouping operators into functional qubit wise commuting groups (along with many others). 

```python
Hamil = Hamil * (1+0.28j)
qwc_bases = Hamil.grouping()
```

Operator multiplications are conducted using the Pauli comutation rules. This allows for computations which may not be classically tractable if one were to represent their Hamiltonian as a matrix.

<h2> ITE </h2>

The ITE ipython notebook details our method to approximate ITE on Quantum Devices (simulated or otherwise). It walks users through the proccess to build a Hamiltonian, run ITE, and compare it to an analytical solution.


***Authors:*** Joe Aulicino, Dr. Bo Peng 

