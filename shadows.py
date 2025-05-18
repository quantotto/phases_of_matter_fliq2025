from typing import List, Tuple
import numpy as np

STATES = {
    "r": np.array([0.0, 1.0]),
    "g": np.array([1.0, 0.0]),
    "+": np.array([1 / np.sqrt(2.0), 1 / np.sqrt(2.0)]),
    "-": np.array([1 / np.sqrt(2.0), -1 / np.sqrt(2.0)]),
    "i": np.array([1 / np.sqrt(2.0), 1.0j * 1 / np.sqrt(2.0)]),
    "-i": np.array([1 / np.sqrt(2.0), -1.0j * 1 / np.sqrt(2.0)]),
}
MAP_LABELS = {"Z2": 0, "Z3": 1}
MAX_SUBSYSTEM_SIZE = 5


def get_one_qubit_density_matrix(state):
    """
    Given a state vector, return the density matrix.
    """
    if state not in STATES:
        raise ValueError(f"Unknown state: {state}")

    vector = STATES[state]
    density_matrix = np.outer(vector, np.conjugate(vector))
    return density_matrix


def sigma_i(state):
    return 3 * get_one_qubit_density_matrix(state) - np.eye(2)


def calculate_reduced_density_matrix(
    measurements: np.array, subsystem: Tuple[int]
) -> np.array:
    """
    Calculate the reduced density matrix for a subsystem.
    """
    T, n = measurements.shape
    if len(subsystem) == 0:
        raise ValueError("Subsystem must not be empty.")
    if len(subsystem) != len(set(subsystem)):
        raise ValueError("Subsystem must not contain repeated indices.")
    if len(subsystem) > min(n, MAX_SUBSYSTEM_SIZE):
        raise ValueError(f"Subsystem size exceeds maximum allowed size.")
    if min(subsystem) < 0 or max(subsystem) >= n:
        raise ValueError("Subsystem indices are out of bounds.")
    density_matrix = np.zeros((2 ** len(subsystem), 2 ** len(subsystem)), dtype=complex)
    for i in range(T):
        product_density = sigma_i(measurements[i, subsystem[0]])
        for j in range(1, len(subsystem)):
            product_density = np.kron(
                product_density, sigma_i(measurements[i, subsystem[j]])
            )
        density_matrix += product_density
    density_matrix /= T
    return density_matrix


def calculate_reduced_vector(measurements: np.array, subsystem: Tuple[int]) -> np.array:
    """
    Calculate the reduced density matrix for a subsystem.
    """
    density_matrices = [
        calculate_reduced_density_matrix(measurements, (subsystem[i],))
        for i in range(len(subsystem))
    ]
    reduced_vector = []
    for density_matrix in density_matrices:
        for num in density_matrix.flatten():
            reduced_vector.append((num * np.conjugate(num)).real)
    return reduced_vector


def generate_subsystem_indices(n: int, k: int) -> List[Tuple[int]]:
    """
    Generate subsequent indices from n qubits.
    """
    if k > n:
        raise ValueError("k must not exceed n.")
    if k < 0:
        raise ValueError("k must not be negative.")
    return [tuple(range(i, i + k)) for i in range(0, n - k + 1, k)]


def create_dataset(
    features: np.array, labels: np.array, subsystem_size: int
) -> Tuple[List[np.array], List[np.array]]:
    """
    Create a dataset of reduced density matrices and their corresponding labels.
    """
    x = []
    y = []
    for i in range(len(features)):
        measurements = features[i]
        for subsystem in generate_subsystem_indices(
            measurements.shape[1], subsystem_size
        ):
            # reduced_density_matrix = calculate_reduced_density_matrix(
            #     measurements, subsystem
            # )
            reduced_vector = calculate_reduced_vector(measurements, subsystem)
            # flattened = []
            # for num in reduced_density_matrix.flatten():
            #     flattened.append(num.real)
            #     flattened.append(num.imag)
            x.append(reduced_vector)
            y.append(MAP_LABELS[labels[i]])
    return x, y
