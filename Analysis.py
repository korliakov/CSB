import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def plot_measurements(L_array, measurements, prob=None):
    """
    Draws a graph of the frequency dependence on L

    Args:
        L_array (numpy.ndarray): array with number of repetitions
        measurements (numpy.ndarray): array with results of measurements
        prob (numpy.ndarray): array with probabilities of being in |0> state


    Returns:
        None
    """

    plt.figure(figsize=(12, 9))
    plt.grid(True)

    if prob is not None:
        plt.plot(L_array, prob, c='blue', label='Theoretical Frequencies')
        plt.fill_between(L_array, prob - np.sqrt(prob * (1 - prob)) / np.sqrt(measurements.shape[1]),
                         prob + np.sqrt(prob * (1 - prob)) / np.sqrt(measurements.shape[1]), color='blue', alpha=0.3,
                         label=r'Theoretical confidence interval')

    plt.plot(L_array, measurements.mean(axis=1), c='r', label='Frequencies from modelling')
    plt.ylim(0, 1)
    plt.xlabel('L')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def get_noisy_eig_ampl(eig_similar, measurements):
    """
    Computes amplitude of noisy eigenvalue

    Args:
        eig_similar (bool): flag of eigenvals similarity
        measurements (np.ndarray): matrix with measurements results

    Returns:
        noisy_eig_ampl (float): amplitude of noisy eigenvalue
    """

    if eig_similar==False:
        pencil_par = 2
    else:
        pencil_par = 1

    prob_est = measurements.mean(axis=1)
    N = len(prob_est)

    Y_1 = np.array([prob_est[p:p + pencil_par] for p in range(0, N - pencil_par)])
    Y_2 = np.array([prob_est[p:p + pencil_par] for p in range(1, N - pencil_par + 1)])

    noisy_eig_ampl = np.abs(np.linalg.eigvals((np.linalg.pinv(Y_1) @ Y_2))).mean()

    return noisy_eig_ampl

def check_eig_similarity(superposition_state_str):
    """
    Checks if eigenvals for states in superposition is similar or not

    Args:
        superposition_state_str (string): string with states

    Returns:
        True if eigenvals are similar, else False

    """

    if '-' in superposition_state_str:
        return False
    else:
        return True





