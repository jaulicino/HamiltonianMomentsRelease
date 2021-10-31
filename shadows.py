'''Helper module for estimating energy with classical shadows'''
import numpy as np
import qiskit as qm

IDENTITY_2 = np.eye(2, dtype=complex)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0., -1.j], [1.j, 0.]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
HADAMARD = (1/np.sqrt(2)) * np.array([[1., 1.], [1., -1.]], dtype=complex)
RX_PI_ON_FOUR = np.array([[1., -1.j], [-1.j, 1.]], dtype=complex)
PAULI_VEC = [PAULI_X, PAULI_Y, PAULI_Z, IDENTITY_2]
PAULI_VEC_LETTERS = {"I": IDENTITY_2, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}

def single_shot_unitary(indeces, circ, backend):
    '''Measure each qubit after applying a unitary (corresponding to indeces[i])'''
    circ_ = circ.copy()
    for i, _ in enumerate(indeces):
        if indeces[i] == 0:
            circ_.h(i)
        elif indeces[i] == 1:
            circ_.rx(np.pi / 2, i)
        elif indeces[i] == 2:
            pass
        circ_.measure(i, i)
    job = qm.execute(circ_, shots=1, backend=backend)
    counts = job.result().get_counts()
    key = list(counts.keys())[0]
    key = key[::-1] # flip for qiskit endianess and our conveince
    key = [-2 * (float(i) - 0.5) for i in key]
    return key


def single_shot_unitary_circuit(indeces, circ):
    '''Measure each qubit after applying a unitary (corresponding to indeces[i])'''
    circ_ = circ.copy()
    for i, _ in enumerate(indeces):
        if indeces[i] == 0:
            circ_.h(i)
        elif indeces[i] == 1:
            circ_.rx(np.pi / 2, i)
        elif indeces[i] == 2:
            pass
        circ_.measure(i, i)
    return circ_


def calculate_classical_shadow(circ, shadow_size, n_qubits, backend):
    '''perform multiple shadow calculations'''
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, n_qubits))
    obs = np.zeros((shadow_size, n_qubits))
    for i in range(shadow_size):
        obs[i, :] = single_shot_unitary(unitary_ids[i], circ, backend)
    return (obs, unitary_ids)


def calculate_classical_shadow_batch(circ, shadow_size, n_qubits, backend):
    '''perform multiple shadow calculations'''
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, n_qubits))
    circuits = []
    obs = np.zeros((shadow_size, n_qubits))
    for i in range(shadow_size):
        circuits += [single_shot_unitary_circuit(unitary_ids[i], circ)]
    job = qm.execute(circuits, backend=backend, shots=1)
    counts = job.result().get_counts()
    for i in range(shadow_size):
        #flip for qiskit endianness
        bits = list(counts[i].keys())[0][::-1]
        obs[i, :] = [-2 * (float(i) - 0.5) for i in bits]
    return (obs, unitary_ids)


def reconstruct_one(snapshot, snapshot_obs):
    '''get the single shot density matrix from the snapshot'''
    rho0 = np.array([[1.0, 0.], [0., 0.]])
    rho1 = np.array([[0.0, 0.], [0., 1.]])

    unitaries = [HADAMARD, RX_PI_ON_FOUR, np.eye(2)]
    rho_estimate = [1]

    for i, _ in enumerate(snapshot):
        state = rho0 if snapshot[i] == 1 else rho1
        unitary = unitaries[snapshot_obs[i]]

        rho_estimate = np.kron(rho_estimate,
                               3 * (np.conjugate(unitary).T @ state @ unitary) - np.eye(2))
    return rho_estimate


def reconstruct(snapshots, snapshots_obs):
    '''Combine and average the snapshots to get an estimate of
    the density operator as a numpy matrix'''
    n_snapshots = len(snapshots)
    n_qbits = len(snapshots[0])
    rho = np.zeros((2**n_qbits, 2**n_qbits), dtype=complex)
    for i in range(n_snapshots):
        rho += reconstruct_one(snapshots[i], snapshots_obs[i])
    return rho / n_snapshots


def operator_norm(diff):
    '''An operator metric for density matrices.'''
    return np.sqrt(np.trace(diff.conjugate().T @ diff))


def obs_to_mat(obs):
    '''Convert an observable to the corresponding matrix'''
    k = 1
    for i, _ in enumerate(obs):
        k = np.kron(k, PAULI_VEC[obs[i]])
    return k


def estimate_shadow_observable(shadow, observable, k=10):
    '''Estimate the expectation value of an observabley,
    with k median of means chunks'''
    snaps, obs = shadow
    nper = int(np.floor(len(obs) / k))
    num_per_max = [nper] * k
    num_per_max[-1] = len(snaps) - nper * (k - 1)
    chunk_means = []
    for i in range(k):
        #each chunk
        sum_ = 0
        for j in range(num_per_max[i]):
            #each part of each chunk
            cobs = obs[i * nper + j]
            contribution = np.zeros(len(cobs))
            for wire, _ in enumerate(cobs):
                #each wire
                if cobs[wire] == observable[wire]:
                    contribution[wire] = 3 * snaps[i * nper + j][wire]
                elif observable[wire] == 3:
                    contribution[wire] = 1.
                else:
                    break
            sum_ += np.prod(contribution)
        chunk_means += [sum_ / num_per_max[i]]

    return np.median(np.array(chunk_means).real)
