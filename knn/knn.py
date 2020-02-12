import numpy as np
from scipy import stats


def main(k: int):
    # dummy data, replace with data from file(s)
    train = np.arange(4 * 4).reshape((4, 4))
    train_labels = np.array(['a', 'b', 'c', 'd'])
    test = np.arange(2 * 4).reshape((2, 4))
    test_labels = np.array(['a', 'b'])

    predicted = knn(k=k, test=test, train=train, train_labels=train_labels)
    accuracy = calculate_accuracy(predicted=predicted, actual=test_labels)

    print(f'Model accuracy was: {accuracy}')


def knn(k: int, test: np.ndarray, train: np.ndarray,
        train_labels: np.ndarray) -> np.ndarray:
    """ Performs the K-nearest neighbors algorithm.

    Args:
        k: int, how many neighbors to consider
        test: np.array of samples to be classified, with
                shape (test_rows, features)
        train: np.array of samples with known labels, with
                shape (train_rows, features)
        train_labels: np.array of labels for the training samples,
                with shape (train_rows, 1)
    Returns:
        np.array with shape (test_rows, 1), indicating the most common label
        of the K nearest neighbors for each testing sample.
    """

    distances = calculate_distances(train=train, test=test)

    # find the K closest training samples
    neighbor_indices = np.argpartition(distances, kth=k, axis=1)[:, :k]

    # get the labels of the K closest training samples
    k_closest_labels = train_labels[neighbor_indices]

    # get the most frequent label of the nearest points
    labels, counts = stats.mode(k_closest_labels, axis=1)
    return labels


def calculate_distances(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """ Calculates the distance between each test point and all training points.

    Args:
        train: np.array of shape (num_train_points, num_features)
        test: np.array of shape (num_test_points, num_features)

    Returns:
        np.array of shape (num_test_points, num_train_points) containing the
        euclidean distance from each test observation to each training
        observation.
    """
    vector_diff = test[:, np.newaxis] - train
    distances = np.linalg.norm(vector_diff, ord=2, axis=2)
    return distances


def calculate_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """ Given two vectors, returns the percentage of values that match between
        the two, element-wise.

    Args:
        predicted: np.array containing the predicted labels
        actual: np.array containing the actual labels

    Returns:
        float, the percentage of predictions that matched the actual label.
    """
    predicted = predicted.reshape((-1,))
    actual = actual.reshape((-1,))
    matches = np.where(predicted == actual, 1, 0)
    return np.sum(matches) / len(matches)


if __name__ == '__main__':
    main(k=2)
