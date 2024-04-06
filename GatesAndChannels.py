from qutip import *
import numpy as np


def Kraus_bit_flip_channel(p):
    """
    List of 2 Kraus operators for bit flip channel

    Args:
        p (float): probability of bit flip

    Returns:
        list: list of 2 Kraus operators for bit flip channel
    """

    return [[qeye(2) * np.sqrt(1 - p), sigmax() * np.sqrt(p)]]


def Kraus_phase_flip_channel(p):
    """
    List of 2 Kraus operators for phase flip channel

    Args:
        p (float): probability of phase flip

    Returns:
        list: list of 2 Kraus operators for phase flip channel
    """

    return [[qeye(2) * np.sqrt(1 - p), sigmaz() * np.sqrt(p)]]


def Kraus_bit_phase_flip_channel(p):
    """
    List of 2 Kraus operators for bit-phase flip channel

    Args:
        p (float): probability of bit-phase flip

    Returns:
        list: list of 2 Kraus operators for bit-phase flip channel
    """

    return [[qeye(2) * np.sqrt(1 - p), sigmay() * np.sqrt(p)]]


def Kraus_depolarizing_channel(p):
    """
    List of 4 Kraus operators for depolarizing channel

    Args:
        p (float): probability of depolarization

    Returns:
        list: list of 4 Kraus operators for depolarizing channel
    """

    return [[qeye(2) * np.sqrt(1 - 0.75 * p), sigmax() * np.sqrt(p) / 2, sigmay() * np.sqrt(p) / 2,
             sigmaz() * np.sqrt(p) / 2]]


def Kraus_amplitude_damping_channel(p):
    """
    List of 2 Kraus operators for amplitude damping channel

    Args:
        p (float): probability of amplitude damping

    Returns:
        list: list of 2 Kraus operators for amplitude damping channel
    """

    return [[Qobj(np.array([[1, 0], [0, np.sqrt(1 - p)]])), Qobj(np.array([[0, np.sqrt(p)], [0, 0]]))]]


def Kraus_phase_damping_channel(p):
    """
    List of 2 Kraus operators for phase damping channel

    Args:
        p (float): probability of phase damping

    Returns:
        list: list of 2 Kraus operators for phase damping channel
    """

    return [[Qobj(np.array([[1, 0], [0, np.sqrt(1 - p)]])), Qobj(np.array([[0, 0], [0, np.sqrt(p)]]))]]


def SWAP_12():
    """
    Returns:
        3-qubit SWAP gate to swap 1 and 2 qubits (acts on 3 qubits)
    """

    return tensor(qeye(2), Qobj(np.array([[1, 0, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 0, 1]]), dims=[[2, 2], [2, 2]]))


def CH():
    """
    Returns:
        2-qubit Controlled H gate (acts on 2 qubits)
    """

    return Qobj(np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                          [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)]]), dims=[[2, 2], [2, 2]])
