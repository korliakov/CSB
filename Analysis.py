import matplotlib.pyplot as plt
import numpy as np


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
