'''helper functions for hamiltonian moments paper'''
from itertools import combinations
import numpy as np
import pennylane as qml

PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0.0, -1.j], [1.j, 0.0]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
IDENTITY_2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
PAULI_DICT = {"I": IDENTITY_2, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}
LETTER_TO_NUMBER = {"X": 1, "Y": 2, "Z": 3}
NUMBER_TO_LETTER = {1: "X", 2: "Y", 3: "Z"}


def pauli_word_to_string(pauli_word, wire_map=None) -> str:
    """
    From:
    https://pennylane.readthedocs.io/en/stable/_modules/pennylane/grouping/utils.html#pauli_word_to_string
    """

    character_map = {
        "Identity": "I",
        "PAULI_X": "X",
        "PAULI_Y": "Y",
        "PAULI_Z": "Z"
    }

    # If there is no wire map, we must infer from the structure of Paulis
    if wire_map is None:
        wire_map = {
            pauli_word.wires.labels[i]: i
            for i, _ in enumerate(pauli_word.wires)
        }

    n_qubits = len(wire_map)

    # Set default value of all characters to identity
    pauli_string = ["I"] * n_qubits

    # Special case is when there is a single Pauli term
    if not isinstance(pauli_word.name, list):
        if pauli_word.name != "Identity":
            wire_idx = wire_map[pauli_word.wires[0]]
            pauli_string[wire_idx] = character_map[pauli_word.name]
        return "".join(pauli_string), [pauli_word.wires[0]]
    for name, wire_label in zip(pauli_word.name, pauli_word.wires):
        wire_idx = wire_map[wire_label]
        pauli_string[wire_idx] = character_map[name]

    return "".join(pauli_string)


def string_to_matrix(letters) -> np.ndarray:
    '''Convert a Pauli string to its matrix representation in the sigma_z basis.'''
    matrix = 1
    for i, _ in enumerate(letters):
        matrix = np.kron(matrix, PAULI_DICT[letters[i]])
    return matrix


def heisenberg_letters(j_constant,
                       u_constant,
                       b_constant,
                       number_sites=4) -> tuple[list[str], list[float]]:
    '''Return the decomposition of a Heisenberg spin Hamiltonian of size n and coefficients,
    J, U, B'''
    letters = []
    coeffs = []
    for comb in list(combinations(list(range(number_sites)), 2)):
        for letter in "XY":
            term_string = ["I"] * number_sites
            term_string[comb[0]] = letter
            term_string[comb[1]] = letter
            term_string = "".join(term_string)
            letters += [term_string]
            coeffs += [j_constant]
        term_string = ["I"] * number_sites
        term_string[comb[0]] = "Z"
        term_string[comb[1]] = "Z"
        term_string = "".join(term_string)
        letters += [term_string]
        coeffs += [u_constant]
    for i in range(number_sites):
        term_string = ["I"] * number_sites
        term_string[i] = "Z"
        term_string = "".join(term_string)
        letters += [term_string]
        coeffs += [b_constant]
    return letters, coeffs


def levi_civita(i, j, k):
    '''Calculate the levi+civita symbol for an i,j,k triplet'''
    return (i != j and i != k and j != k) * ((i < j) * 2 - 1) * (
        (j < k) * 2 - 1) * ((i < k) * 2 - 1)


def pq_merge(string1, string2) -> tuple[str, float]:
    '''Multiply two Pauli strings'''
    letter_to_number = {"X": 1, "Y": 2, "Z": 3}
    number_to_letter = {1: "X", 2: "Y", 3: "Z"}
    assert len(string1) == len(string2)
    if string1 == string2:
        return "I" * len(string1), 1
    product_letters = []
    sgn = 1
    for letter, _ in enumerate(string1):
        if string1[letter] == "I":
            product_letters += [string2[letter]]
        elif string2[letter] == "I":
            product_letters += [string1[letter]]
        elif string1[letter] == string2[letter]:
            product_letters += ["I"]
        else:
            i = letter_to_number[string1[letter]]
            j = letter_to_number[string2[letter]]
            k = 6 - i - j
            product_letters += [number_to_letter[k]]
            sgn *= levi_civita(i, j, k) * 1.j
    return "".join(product_letters), sgn



def mergn(letters) -> tuple[str, float]:
    '''Multiply multiple Pauli words'''
    if len(letters) == 2:
        return pq_merge(letters[0], letters[1])
    pair_letters, pair_sign = pq_merge(letters[0], letters[1])
    next_letters, next_sign = mergn([pair_letters] + letters[2:])
    return next_letters, pair_sign * next_sign


def poly(hamiltonian_matrix, coefficient_list) -> np.ndarray:
    '''Polynomial expansion for matrix given coefficients'''
    expansion = np.eye(len(hamiltonian_matrix), dtype=complex)
    for i, _ in enumerate(coefficient_list):
        expansion += np.linalg.matrix_power(hamiltonian_matrix,
                                            i + 1) * coefficient_list[i]
    return expansion


def pade_expansion(taylor_order, pade_order) -> tuple[np.ndarray, np.ndarray]:
    '''Compute Pade(P,string2) for polynomial with coefficients (1,c1,c2,...,ck)'''
    coeff = np.array(
        [1 / np.math.factorial(i + 1) for i in range(taylor_order)],
        dtype="float")
    expansion = np.zeros((taylor_order, taylor_order))
    expansion[:pade_order, :pade_order] = np.eye(pade_order)
    row = 0
    for i in range(pade_order, taylor_order):
        expansion[row, i] = -1.0
        row += 1
        expansion[row:, i] = -1.0 * np.copy(coeff[:(taylor_order - row)])
    x_solution = np.linalg.solve(expansion, coeff)
    return x_solution[:pade_order], x_solution[pade_order:]


def are_qwc(string1, string2) -> bool:
    '''Check to see if two Pauli strings are qubit wise commuting.'''
    for i, _ in enumerate(string1):
        if not (string1[i] == "I" or string2[i] == "I"
                or string1[i] == string2[i]):
            return False
    return True


def qwc_pairs(terms) -> list[tuple]:
    '''Generate pairs of qwc strings from a list of terms.'''
    pairs = []
    for i, _ in enumerate(terms):
        for j in range(i + 1):
            if i == j:
                pairs += [[terms[i], terms[j]]]
            elif are_qwc(terms[i], terms[j]):
                pairs += [[terms[i], terms[j]]]
    return pairs


def measurement_basis(groups) -> list[str]:
    '''Find the measurement bases from a dictionary with the given groups.'''
    bases = []
    for i in range((max(list(groups.values())) + 1)):
        bases += [["I"] * len(list(groups.keys())[0])]
    for string in groups:
        for i, _ in enumerate(string):
            if bases[groups[string]][i] == "I":
                bases[groups[string]][i] = string[i]
    return ["".join(base) for base in bases]
