'''Hamiltonian class |
   requires numpy, itertools, math, networkx and helper.py

'''
from __future__ import annotations
import math
import itertools
import numpy as np
import Hamiltonians.helper as helper
import networkx as nx


class Hamiltonian():
    '''
        The Hamiltonian class provides a paradigm to perform manipulations on
        Hamiltonians. The class operates using the multiplication rules of
        the Pauli group. This allows for computations which may not be possible
        if one were to represent their Hamiltonian as a matrix

        Attributes:
            letters: a list of Pauli strings
            coeffs: coefficients of specified Pauli strings,
                    definied in the same order as their corresponding string
    '''
    def __init__(self, letters, coeffs) -> Hamiltonian:
        '''Initialize a Hamiltonian. '''
        self.letters = letters
        self.coeffs = coeffs

    def condense(self) -> Hamiltonian:
        '''Combine duplicate pauli words and their coefficients and remove terms
        with coefficient zero.
        '''
        dict_ = {}
        for i in range(len(self.letters)):
            if self.letters[i] in dict_:
                dict_[self.letters[i]] += self.coeffs[i]
            else:
                dict_[self.letters[i]] = self.coeffs[i]
        keys_to_pop = []
        for key in dict_:
            if np.abs(dict_[key]) == 0:
                keys_to_pop += [key]
        for i in keys_to_pop:
            dict_.pop(i)
        letters = list(dict_.keys())
        values = list(dict_.values())
        return Hamiltonian(letters, values)

    def multiply(self, multiplier_hamiltonian) -> Hamiltonian:
        #  Multiply one hamiltonian by another.
        product_dict = {}
        for i in range(len(self.letters)):
            for j in range(len(multiplier_hamiltonian.letters)):
                prod, sgn = helper.pq_merge(self.letters[i],
                                            multiplier_hamiltonian.letters[j])
                coeff = self.coeffs[i] * multiplier_hamiltonian.coeffs[j] * sgn
                if not prod in product_dict:
                    product_dict[prod] = 0
                product_dict[prod] += coeff
        return Hamiltonian(list(product_dict.keys()),
                           list(product_dict.values())).condense().clean()

    def power(self, exponent) -> Hamiltonian:
        '''Raise self Hamiltonian to power and condense resulting Hamiltonian'''
        if exponent == 0:
            return Hamiltonian(["I" * len(self.letters[0])], [1.0])
        elif exponent == 1:
            return self
        return self.multiply(self).multiply(self.power(exponent - 2))

    def mult_scalar(self, complex_scalar) -> Hamiltonian:
        #  Multiply a Hamiltonian by a complex scalar.
        cprime = [i * complex_scalar for i in self.coeffs]
        return Hamiltonian(self.letters, cprime)

    def add(self, second_hamiltonian) -> Hamiltonian:
        '''Add two Hamiltonians.'''
        letters = self.letters + second_hamiltonian.letters
        coeffs = self.coeffs + second_hamiltonian.coeffs
        return Hamiltonian(letters, coeffs).condense()

    def __add__(self, second_hamiltonian) -> Hamiltonian:
        #  Support use of '+'.
        return self.add(second_hamiltonian)

    def __mul__(self, multiplier) -> Hamiltonian:
        #  Support use of '*'.
        if isinstance(multiplier, Hamiltonian):
            return self.multiply(multiplier)
        return self.mult_scalar(multiplier)

    def exp(self, order=3) -> Hamiltonian:
        '''Approximate a Hamiltonian exponential with an nth order Taylor expansion.'''
        sum_hamiltonian = Hamiltonian(["I" * len(self.letters[0])], [1.0])
        for i in range(1, order):
            power_hamiltonian = self.power(i).multScalar(1 /
                                                         (math.factorial(i)))
            sum_hamiltonian = sum_hamiltonian.add(power_hamiltonian)
            sum_hamiltonian = sum_hamiltonian.condense()
            sum_hamiltonian = sum_hamiltonian.clean()
        return sum_hamiltonian

    def to_matrix(self) -> np.ndarray:
        '''Convert Hamiltonian to a numpy matrix.'''
        size = int(len(self.letters[0]))
        arr = np.zeros((2**size, 2**size), dtype=complex)
        for i in range(len(self.letters)):
            operator_matrix = helper.string_to_matrix(self.letters[i])
            arr += self.coeffs[i] * operator_matrix
        return arr

    def clean(self, tol=1e8) -> Hamiltonian:
        '''Remove Pauli strings which have coefficients of relatively low magnitude
        based on a supplied tolerance.
        '''
        max_ = np.max(np.abs(self.coeffs))
        non_dels = []
        for i in range(len(self.coeffs)):
            if max_ / np.abs(self.coeffs[i]) > tol:
                pass
            else:
                non_dels += [i]
        clean_coeffs = list(np.array(self.coeffs)[non_dels])
        clean_letters = list(np.array(self.letters)[non_dels])
        return Hamiltonian(clean_letters, clean_coeffs)

    def grouping_exact(self) -> list[str]:
        '''Exactly partition hamiltonian terms in QWC groups to reduce the number of measurements
        required.
        '''
        nodes = self.letters
        edges = helper.qwc_pairs(nodes)
        qwc_graph = nx.Graph()
        qwc_graph.add_nodes_from(nodes)
        qwc_graph.add_edges_from(edges)
        complement_graph = nx.complement(qwc_graph)
        groups = nx.coloring.greedy_color(complement_graph,
                                          strategy="largest_first")
        bases = helper.measurement_basis(groups)
        return bases

    def grouping(self, size=500, tried=-1) -> Hamiltonian:
        '''Perform a grouping scheme to reduce the number of measurments required.
        For Hamiltonians with a number of terms greater than 'size,' partition the
        Hamiltonian into managable chunks.
        '''
        nterms = len(self.letters)
        bases = []
        if nterms <= size:
            return self.grouping_exact()
        else:
            #  Partition the Hamiltonian if it exceed the size constraint.
            num_hamils = int(np.ceil(nterms / size))
            for i in range(num_hamils - 1):
                bases += Hamiltonian(self.letters[i * size:(i + 1) * size],
                                     self.coeffs[i * size:(i + 1) *
                                                 size]).grouping_exact()
            #  Deal with the last chunk which may not have the desired size.
            bases += Hamiltonian(self.letters[(i + 1) * size:],
                                 self.coeffs[(i + 1) *
                                             size:]).grouping_exact()
        #  Repeat grouping scheme on current list of QWC bases until the length
        #  of the bases converges.
        qwc_bases_ham = Hamiltonian(bases, np.ones(len(bases))).condense()
        current_QWC_bases_length = len(bases)
        if current_QWC_bases_length < size:
            return qwc_bases_ham.grouping_exact()
        else:
            #  tried == -1 when scheme is first executed for a given Hamiltonian.
            if tried == -1:

                # Test to see if a second grouping reduces the size of the set of qwc bases.
                # If it does not, return the current set. If it does, try grouping again.

                next_run = qwc_bases_ham.grouping(size=size, tried=current_QWC_bases_length)
                if current_QWC_bases_length <= len(next_run):
                    return bases
                return Hamiltonian(next_run, np.ones(len(next_run))).grouping(
                    size=size, tried=len(next_run))
            else:

                # If this round of grouping did not reduce the number of QWC bases
                # return the current set of qwc bases. Otherwise, continue until convergence.

                if tried <= current_QWC_bases_length:
                    return bases
                return qwc_bases_ham.grouping(size=size, tried=len(bases))

    def estimate_hartree_fock(self, nelec) -> (np.ndarray, float):
        '''Estimate the Hartree-Fock state for a Hamiltonian and a given number of electrons'''
        num_qubits = len(self.letters[0])
        ham_matrix = self.to_matrix()
        _bit = []
        for i in range(0, num_qubits):
            _bit.extend([i])
        _pool = list(itertools.combinations(_bit, nelec))
        cnt = 0
        ind = 0
        for k in _pool:
            #
            state = 1.0
            for key in _bit:
                if key in k:
                    vtmp = np.array([0., 1.], dtype=complex)
                else:
                    vtmp = np.array([1., 0.], dtype=complex)
                state = np.kron(vtmp, state)
            energy = (np.conj(state).T @ ham_matrix @ state).real
            if cnt == 0:
                e_min = np.copy(energy)
            else:
                if energy < e_min:
                    ind = np.copy(cnt)
                    e_min = np.copy(energy)
            cnt += 1
        return np.array(_pool[ind]), e_min
