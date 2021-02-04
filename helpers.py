from constants import ALGORITHM_LABELS, N_FOLDS


def get_algorithms():
    algorithms = []

    for key, label in ALGORITHM_LABELS.items():
        for fold in range(N_FOLDS):
            algorithms.append({
                "id": key,
                "label": label,
                "fold": fold,
            })

    return algorithms
