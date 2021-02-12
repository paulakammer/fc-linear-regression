import numpy as np


def aggregate_offsets(offsets, samples_per_client, overall_sample_size):
    weighted_X_offsets = offsets[0] * samples_per_client[0]
    for i in range(1, len(offsets)):
        weighted_X_offsets = weighted_X_offsets + (offsets[i] * samples_per_client[i])
    X_offset_global = weighted_X_offsets / overall_sample_size

    return X_offset_global


def aggregate_matrices(matrices):
    matrix = matrices[0]
    for i in range(1, len(matrices)):
        matrix = np.add(matrix, matrices[i])
    matrix_global = matrix

    return matrix_global


def aggregate_preprocessing(preprocessing):
    X_offsets = [client[0] for client in preprocessing]
    X_offsets = [np.array(el).astype(np.float) if type(el) is not np.ndarray else el for el in X_offsets]
    y_offsets = [client[1] for client in preprocessing]
    y_offsets = [np.array(el).astype(np.float) if type(el) is not np.ndarray else el for el in y_offsets]
    X_scales = [client[2] for client in preprocessing]
    X_scales = [np.array(el).astype(np.float) if type(el) is not np.ndarray else el for el in X_scales]

    samples_per_client = [client[3] for client in preprocessing]
    samples_per_client = [int(el) for el in samples_per_client]
    overall_sample_size = np.sum(samples_per_client)
    X_offset_global = aggregate_offsets(X_offsets, samples_per_client, overall_sample_size)
    y_offset_global = aggregate_offsets(y_offsets, samples_per_client, overall_sample_size)
    X_scale_global = aggregate_offsets(X_scales, samples_per_client, overall_sample_size)

    return X_offset_global, y_offset_global, X_scale_global


def aggregate_beta(local_results):
    XT_X_matrices = [np.array(client[0]) for client in local_results]
    XT_X_matrix_global = aggregate_matrices(XT_X_matrices)

    XT_y_vectors = [np.array(client[1]) for client in local_results]
    XT_y_vector_global = aggregate_matrices(XT_y_vectors)

    XT_X_matrix_inverse = np.linalg.inv(XT_X_matrix_global)
    beta_vector = np.dot(XT_X_matrix_inverse, XT_y_vector_global)

    return beta_vector





