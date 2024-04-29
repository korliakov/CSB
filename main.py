from Analysis import *
from Circuits import *
from GatesAndChannels import *
from IonBackend import *
import matplotlib.pyplot as plt


def main():
    fidelity, noisy_eig_array_nonsim, noisy_eig_array_sim = full_toffoli_experiment([[qeye(2)]], [*Kraus_amplitude_damping_channel(0.01), *Kraus_depolarizing_channel(0.01), *Kraus_phase_damping_channel(0.01)], L_max=10, N_s=1000, K=10)
    print(fidelity)

if __name__ == "__main__":
    main()
