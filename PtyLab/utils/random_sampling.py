import numpy as np

def choose_with_arbitrary_histogram(N, probabilities):
    """

    Choose numbers from 1 to N with a given probability distribution

    Parameters
    ----------
    N: int
        Number of points to return
    probabilities: np.ndarray of length M
        Relative probability for every interval.

    Returns
    -------

    """
    bins = probabilities.cumsum()
    bins = bins / bins.max()
    random_numbers = np.random.rand(N)
    return np.argmin(random_numbers[:,None] >= bins[None,:], axis=1)
