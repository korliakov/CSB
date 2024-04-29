from qutip import *
from qutip.qip.operations import snot, cnot, rx, ry, rz, molmer_sorensen, toffoli
import numpy as np

from GatesAndChannels import CH


def X_gate_repr(idx):
    """
    X gate in ion native gates

    Args:
        idx (list): idx (one element list) for X gate to act on

    Returns:
        (list(tuple(list(list(qutip.qobj.Qobj)), list()))): subcircuit description of X gate using native ion gates
    """

    return [([[rx(np.pi)]], idx)]


def Y_gate_repr(idx):
    """
    Y gate in ion native gates

    Args:
        idx (list): idx (one element list) for Y gate to act on

    Returns:
        (list(tuple(list(list(qutip.qobj.Qobj)), list()))): subcircuit description of Y gate using native ion gates
    """

    return [([[ry(np.pi)]], idx)]


def Z_gate_repr(idx):
    """
    Z gate in ion native gates

    Args:
        idx (list): idx (one element list) for Z gate to act on

    Returns:
        (list(tuple(list(list(qutip.qobj.Qobj)), list()))): subcircuit description of Z gate using native ion gates
    """

    return [([[ry(np.pi / 2)]], idx), ([[rx(np.pi)]], idx), ([[ry(-np.pi / 2)]], idx)]


def H_gate_repr(idx):
    """
    H gate in ion native gates

    Args:
        idx (list): idx (one element list) for H gate to act on

    Returns:
        (list(tuple(list(list(qutip.qobj.Qobj)), list()))): subcircuit description of H gate using native ion gates
    """

    return [([[rx(np.pi)]], idx), ([[ry(-np.pi / 2)]], idx)]


def Cnot_gate_repr(idx):
    """
    Cnot gate in ion native gates

    Args:
        idx (list): idx (two element list) for Cnot gate to act on

    Returns:
        (list(tuple(list(list(qutip.qobj.Qobj)), list()))): subcircuit description of Cnot gate using native ion gates
    """

    return [([[ry(np.pi / 2)]], [idx[0]]), ([[molmer_sorensen(np.pi / 2)]], idx), ([[rx(-np.pi / 2)]], [idx[0]]),
            ([[rx(-np.pi / 2)]], [idx[1]]), ([[ry(-np.pi / 2)]], [idx[0]])]


def CH_gate_repr(idx):
    """
    CH gate in ion native gates

    Args:
        idx (list): idx (two element list) for CH gate to act on

    Returns:
        (list(tuple(list(list(qutip.qobj.Qobj)), list()))): subcircuit description of CH gate using native ion gates
    """

    return ([[ry(np.pi / 4)]], [idx[1]]), ([[cnot()]], idx), ([[ry(-np.pi / 4)]], [idx[1]])


def Toff_gate_repr():
    """
    Toff gate in ion native gates

    Returns:
        (list(tuple(list(list(qutip.qobj.Qobj)), list()))): subcircuit description of Toff gate using native ion gates
    """

    return [*H_gate_repr([2]), *Cnot_gate_repr([1, 2]), ([[rz(-np.pi / 4)]], [2]), *Cnot_gate_repr([0, 2]),
            ([[rz(np.pi / 4)]], [2]),
            *Cnot_gate_repr([1, 2]), ([[rz(-np.pi / 4)]], [2]), *Cnot_gate_repr([0, 2]), ([[rz(np.pi / 4)]], [1]),
            ([[rz(np.pi / 4)]], [2]),
            *Cnot_gate_repr([0, 1]), *H_gate_repr([2]), ([[rz(np.pi / 4)]], [0]), ([[rz(-np.pi / 4)]], [1]),
            *Cnot_gate_repr([0, 1])]


def get_ion_subcircuit_description(subcircuit_description):
    """
    Makes transpiled version of subcircuit_description with ion native gates

    Args:
        subcircuit_description (list(tuple(list(list(qutip.qobj.Qobj)), list()))): subcircuit description with widely used gates

    Returns:
        ion_subcircuit (list(tuple(list(list(qutip.qobj.Qobj)), list()))): subcircuit description with ion native gates
    """

    ion_subcircuit = []
    for gate_description in subcircuit_description:
        gate = gate_description[0][0][0]
        idx = gate_description[1]
        if len(idx) == 1:
            if gate == sigmax():
                ion_subcircuit.extend(X_gate_repr(idx))
            elif gate == sigmay():
                ion_subcircuit.extend(Y_gate_repr(idx))
            elif gate == sigmaz():
                ion_subcircuit.extend(Z_gate_repr(idx))
            elif gate == snot():
                ion_subcircuit.extend(H_gate_repr(idx))
            else:
                raise Exception("Unknown 1-qubit gate")
        elif len(idx) == 2:
            if gate == cnot():
                ion_subcircuit.extend(Cnot_gate_repr(idx))
            elif gate == CH():
                ion_subcircuit.extend(CH_gate_repr(idx))
            else:
                raise Exception("Unknown 2-qubit gate")
        elif len(idx) == 3:
            if gate == toffoli():
                ion_subcircuit.extend([gate_description])
            else:
                raise Exception("Unknown 3-qubit gate")
    return ion_subcircuit
