from qutip.qip.operations import snot
import random

from Circuits import *
from GatesAndChannels import *
from Analysis import *


def main():
    channel_to_repeat_description = [([[tensor(snot(), snot())]], [0, 1])]
    unitary_gate_to_repeat = three_qubit_circuit_assembler(channel_to_repeat_description, [[qeye(2)]])[0][0]
    c1 = random.random() + 1j * random.random()
    c2 = random.random() + 1j * random.random()
    psi_prep = c1 * random.choice(unitary_gate_to_repeat.eigenstates()[1]) + c2 * random.choice(
        unitary_gate_to_repeat.eigenstates()[1])
    psi_prep = psi_prep / psi_prep.norm()

    U_prep_gate = get_prep_gate(psi_prep)
    U_prep_description = [([[U_prep_gate]], [1, 2, 3])]
    U_meas_description = [([[U_prep_gate.dag()]], [1, 2, 3])]
    list_of_one_qubit_noises = [*Kraus_amplitude_damping_channel(0.01), *Kraus_phase_damping_channel(0.01),
                                *Kraus_depolarizing_channel(0.01)]

    L_array, measurements, prob = quantum_circuit_with_L_reps(U_prep_description, channel_to_repeat_description,
                                                              U_meas_description, list_of_one_qubit_noises, L_max=100,
                                                              N_s=1000)

    plot_measurements(L_array, measurements, prob)


if __name__ == "__main__":
    main()
