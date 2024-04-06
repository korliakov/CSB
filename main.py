from Analysis import *
from Circuits import *
from GatesAndChannels import *
from IonBackend import *


def main():
    U_prep_description = get_Toff_superposition_preparation_subcircuit_dict()['11+,11-']
    U_prep_description_ion = get_ion_subcircuit_description(U_prep_description)
    U_meas_description_ion = get_dagger_subcircuit(U_prep_description_ion)

    gate_to_repeat_description = [([[toffoli()]], [0, 1, 2])]
    gate_to_repeat_description_ion = get_ion_subcircuit_description(gate_to_repeat_description)

    list_of_one_qubit_noises = [*Kraus_amplitude_damping_channel(0.001), *Kraus_depolarizing_channel(0.001),
                                *Kraus_phase_damping_channel(0.001)]

    L_array, measurements, prob = quantum_circuit_with_L_reps(U_prep_description_ion, gate_to_repeat_description_ion,
                                                              U_meas_description_ion, list_of_one_qubit_noises,
                                                              L_max=20,
                                                              N_s=1000)

    plot_measurements(L_array, measurements, prob)


if __name__ == "__main__":
    main()
