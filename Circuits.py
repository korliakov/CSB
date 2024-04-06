import numpy as np
from qutip import *
from qutip.qip.operations import snot, cnot
from tqdm import tqdm

from GatesAndChannels import SWAP_12, CH


def quantum_circuit(initial_dm, list_of_channels):
    """
    Models quantum circuit with quantum channels and unitary gates

    Args:
        initial_dm (qutip.qobj.Qobj): initial density matrix in matrix form
        list_of_channels (list(list(qutip.qobj.Qobj))): list of lists with Kraus operators for each quantum channel

    Returns:
        qutip.qobj.Qobj: final density matrix in matrix form
    """

    vec_dm = operator_to_vector(initial_dm)
    for channel in list_of_channels:
        vec_dm = sum(list(map(to_super, channel))) * vec_dm
    return vector_to_operator(vec_dm)


def get_three_qubit_subcircuit(qubit, one_qubit_subcircuit):
    """
    Makes subcircuit acting on 3 qubits form 1 qubit subcircuit

    Args:
        qubit (int): the number of the qubit affected by the noise
        one_qubit_subcircuit (list(list(qutip.qobj.Qobj))): one qubit subcircuit

    Returns:
        three_qubit_subcircuit (list(list(qutip.qobj.Qobj))): three qubit subcircuit
    """
    if qubit == 0:
        tensor_prod_func = lambda x: tensor(x, qeye(2), qeye(2))
    elif qubit == 1:
        tensor_prod_func = lambda x: tensor(qeye(2), x, qeye(2))
    else:
        tensor_prod_func = lambda x: tensor(qeye(2), qeye(2), x)

    three_qubit_subcircuit = []

    for noise in one_qubit_subcircuit:
        three_qubit_subcircuit.extend([list(map(tensor_prod_func, noise))])
    return three_qubit_subcircuit


def three_qubit_circuit_assembler(circuit_description, list_of_one_qubit_noises):
    """
    Assembles a circuit for 3 qubits

    Args:
        circuit_description (list(tuple(list(list(qutip.qobj.Qobj)), list()))): list of 1, 2 and 3 qubit gates with qubit/qubits idx
        list_of_one_qubit_noises (list(list(qutip.qobj.Qobj))): list of Kraus operators for each noise type

    Returns:
        list_of_channels (list(list(qutip.qobj.Qobj))): list of lists with Kraus operators and unitary operations for each quantum channel
    """

    list_of_channels = []

    for gate, qubit_idx in circuit_description:
        if len(qubit_idx) == 1:
            channel_with_noise = gate.copy()
            channel_with_noise.extend(list_of_one_qubit_noises)
            list_of_channels.extend(get_three_qubit_subcircuit(qubit_idx[0], channel_with_noise))
        elif len(qubit_idx) == 2:
            if qubit_idx == [0, 1]:
                list_of_channels.extend([[tensor(gate[0][0], qeye(2))]])
                list_of_channels.extend(get_three_qubit_subcircuit(qubit_idx[0], list_of_one_qubit_noises))
                list_of_channels.extend(get_three_qubit_subcircuit(qubit_idx[1], list_of_one_qubit_noises))
            elif qubit_idx == [1, 2]:
                list_of_channels.extend([[tensor(qeye(2), gate[0][0])]])
                list_of_channels.extend(get_three_qubit_subcircuit(qubit_idx[0], list_of_one_qubit_noises))
                list_of_channels.extend(get_three_qubit_subcircuit(qubit_idx[1], list_of_one_qubit_noises))
            elif qubit_idx == [0, 2]:
                list_of_channels.extend([[SWAP_12() * tensor(gate[0][0], qeye(2)) * SWAP_12()]])
                list_of_channels.extend(get_three_qubit_subcircuit(qubit_idx[0], list_of_one_qubit_noises))
                list_of_channels.extend(get_three_qubit_subcircuit(qubit_idx[1], list_of_one_qubit_noises))
        elif len(qubit_idx) == 3:
            list_of_channels.extend(gate)
            list_of_channels.extend(get_three_qubit_subcircuit(qubit_idx[0], list_of_one_qubit_noises))
            list_of_channels.extend(get_three_qubit_subcircuit(qubit_idx[1], list_of_one_qubit_noises))
            list_of_channels.extend(get_three_qubit_subcircuit(qubit_idx[2], list_of_one_qubit_noises))

    return list_of_channels


def quantum_circuit_with_L_reps(U_prep_description, channel_to_repeat_description, U_meas_description,
                                list_of_one_qubit_noises,
                                L_max=100, N_s=100):
    """
    Models quantum circuits with a repeating quantum channel

    Args:
        U_prep_description (list(tuple(list(list(qutip.qobj.Qobj)), list()))): preparation subcircuit description
        channel_to_repeat_description (list(tuple(list(list(qutip.qobj.Qobj)), list()))): channel to repeat subcircuit description
        U_meas_description (list(tuple(list(list(qutip.qobj.Qobj)), list()))): measurement subcircuit description
        list_of_one_qubit_noises (list(list(qutip.qobj.Qobj))): list of Kraus operators for each noise type
        L_max (int): the maximum number of repetitions of the quantum channel
        N_s (int): number of shots

    Returns:
        prob (numpy.ndarray): array with probabilities of being in |0> state
        measurements (numpy.ndarray): array with results of measurements
    """
    initial_dm = tensor(fock_dm(2, 0), fock_dm(2, 0), fock_dm(2, 0))
    L_array = np.arange(0, L_max + 1)
    prob = np.zeros(L_max + 1)
    measurements = np.zeros((L_max + 1, N_s))

    basis_state = tensor(fock_dm(2, 0), fock_dm(2, 0), fock_dm(2, 0))

    for L in tqdm(L_array):
        circuit_list = []
        circuit_list.extend(three_qubit_circuit_assembler(U_prep_description, list_of_one_qubit_noises))
        for i in range(L):
            circuit_list.extend(three_qubit_circuit_assembler(channel_to_repeat_description, list_of_one_qubit_noises))
        circuit_list.extend(three_qubit_circuit_assembler(U_meas_description, list_of_one_qubit_noises))

        final_dm = quantum_circuit(initial_dm, circuit_list)

        prob[L] = abs(((final_dm * basis_state).tr())) ** 2

        measurements[L] = np.random.binomial(1, np.min((abs(((final_dm * basis_state).tr())) ** 2, 1.0)), N_s)

    return L_array, measurements, prob


def get_prep_gate(desired_ket):
    """
    Constructs preparation gate to prepare desired state from |0>

    Args:
        desired_ket (qutip.qobj.Qobj(dims=[[2, 2, 2], [1, 1, 1]])): Desired ket vector

    Returns:
        Preparation gate (qutip.qobj.Qobj(dims=[[2, 2, 2], [2, 2, 2]]))
    """

    M = np.zeros((8, 8), dtype='complex_')
    M[:, 0] = np.array(desired_ket).reshape(1, -1)
    Q, R = np.linalg.qr(M)

    return (-1) * Qobj(Q, dims=[[2, 2, 2], [2, 2, 2]])


def get_Toff_superposition_preparation_subcircuit_dict():
    """
    Dict with equal superposition of eigenstates of Toff gate

    Returns:
        prep_dict (dict): keys - strings with states, values - subcircuit description of preparation subcircuit
    """

    prep_dict = {'000,001': [([[snot()]], [2])],
                 '010,011': [([[sigmax()]], [1]), ([[snot()]], [2])],
                 '100,101': [([[sigmax()]], [0]), ([[snot()]], [2])],
                 '11+,11-': [([[sigmax()]], [0]), ([[sigmax()]], [1])],
                 '000,010': [([[snot()]], [1])],
                 '001,101': [([[snot()]], [0]), ([[sigmax()]], [2])],
                 '000,100': [([[snot()]], [0])],
                 '001,011': [([[snot()]], [1]), ([[sigmax()]], [2])],
                 '010,100': [([[snot()]], [0]), ([[sigmax()]], [0]), ([[cnot()]], [0, 1]), ([[sigmax()]], [0])],
                 '011,101': [([[snot()]], [0]), ([[sigmax()]], [0]), ([[cnot()]], [0, 1]), ([[sigmax()]], [0]),
                             ([[sigmax()]], [2])],
                 '000,011': [([[snot()]], [1]), ([[cnot()]], [1, 2])],
                 '010,001': [([[snot()]], [1]), ([[sigmax()]], [1]), ([[cnot()]], [1, 2]), ([[sigmax()]], [1])],
                 '000,101': [([[snot()]], [0]), ([[cnot()]], [0, 2])],
                 '001,100': [([[snot()]], [0]), ([[sigmax()]], [0]), ([[cnot()]], [0, 2]), ([[sigmax()]], [0])],
                 '010,101': [([[snot()]], [0]), ([[cnot()]], [0, 1]), ([[cnot()]], [0, 2]), ([[sigmax()]], [1])],
                 '011,100': [([[snot()]], [0]), ([[cnot()]], [0, 1]), ([[cnot()]], [0, 2]), ([[sigmax()]], [0])],
                 '11+,000': [([[snot()]], [0]), ([[cnot()]], [0, 1]), ([[CH()]], [1, 2])],
                 '11+,001': [([[snot()]], [0]), ([[cnot()]], [0, 1]), ([[CH()]], [1, 2]), ([[sigmax()]], [2])],
                 '11+,010': [([[snot()]], [0]), ([[sigmax()]], [1]), ([[CH()]], [0, 2])],
                 '11+,011': [([[snot()]], [0]), ([[sigmax()]], [1]), ([[sigmax()]], [0]), ([[cnot()]], [0, 2]),
                             ([[sigmax()]], [0]), ([[CH()]], [0, 2])],
                 '11+,100': [([[sigmax()]], [0]), ([[snot()]], [1]), ([[CH()]], [1, 2])],
                 '11+,101': [([[sigmax()]], [0]), ([[snot()]], [1]), ([[sigmax()]], [1]), ([[cnot()]], [1, 2]),
                             ([[sigmax()]], [1]), ([[CH()]], [1, 2])],
                 '11-,000': [([[snot()]], [0]), ([[cnot()]], [0, 1]), ([[CH()]], [1, 2]), ([[sigmaz()]], [2])],
                 '11-,001': [([[snot()]], [0]), ([[sigmax()]], [2]), ([[cnot()]], [0, 1]), ([[CH()]], [1, 2])],
                 '11-,010': [([[snot()]], [0]), ([[sigmax()]], [1]), ([[cnot()]], [0, 2]), ([[CH()]], [0, 2])],
                 '11-,011': [([[snot()]], [0]), ([[sigmax()]], [1]), ([[sigmax()]], [2]), ([[CH()]], [0, 2])],
                 '11-,100': [([[sigmax()]], [0]), ([[snot()]], [1]), ([[cnot()]], [1, 2]), ([[CH()]], [1, 2])],
                 '11-,101': [([[sigmax()]], [0]), ([[snot()]], [1]), ([[sigmax()]], [2]), ([[CH()]], [1, 2])]}

    return prep_dict


def get_dagger_subcircuit(subcircuit_description):
    """
    Function to make subcircuit_description.dag(). Main purpose to get measurement subcircuit from preparation subcircuit

    Args:
        subcircuit_description (list(tuple(list(list(qutip.qobj.Qobj)), list()))): description of subcircuit to make .dag()

    Returns:
        subcircuit_dag[::-1] (list(tuple(list(list(qutip.qobj.Qobj)), list()))): subcircuit_description.dag()
    """
    subcircuit_dag = []
    for gate_description in subcircuit_description:
        gate = gate_description[0][0][0]
        idx = gate_description[1]
        subcircuit_dag.append(([[gate.dag()]], idx))
    return subcircuit_dag[::-1]
