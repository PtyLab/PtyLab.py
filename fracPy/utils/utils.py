import pickle



def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
