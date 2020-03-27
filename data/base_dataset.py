class BaseDataset:
    def __init__(self):
        pass

    def get_k_samples(self, idx, k, mode, cls=None, shuffle=True):
        raise NotImplementedError
