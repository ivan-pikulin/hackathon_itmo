import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader


def train_test_valid_split(dataset, n_splits=5, test_ratio=0.2, batch_size=64):
    """
    Makes KFold cross-validation

    Parameters
    ----------
    dataset : Dataset
    n_splits : int, optional
        Number of folds in cross-validatoin
    test_ratio : float from 0.0 to 1.0, optional
        Percentage of test data in dataset
    batch_size : int, optional

    Returns
    -------
    folds : list
        List of cross-validation folds in format (train_loader, valid_loader)
    test_loader (if return_test) : DataLoader
        Test DataLoader, which does not participate in cross-validation
    """
    dataset_size = len(dataset)
    ids = range(dataset_size)
    if test_ratio > 0:
        train_ids, test_ids = train_test_split(ids, test_size=test_ratio, random_state=14)
        test_loader = DataLoader([val for i, val in enumerate(dataset) if i in test_ids],
                                 batch_size=batch_size)
    else:
        train_ids = ids

    folds = []
    kf_split = KFold(n_splits=n_splits)
    for train_index, valid_index in kf_split.split(train_ids):
        train_loader = DataLoader([val for i, val in enumerate(dataset) if i in train_index], batch_size=batch_size)
        valid_loader = DataLoader([val for i, val in enumerate(dataset) if i in valid_index], batch_size=batch_size)
        folds += [(train_loader, valid_loader)]
    if test_ratio > 0:
        return folds, test_loader
    else:
        return folds


def get_example_dataset():
    from sklearn.datasets import load_iris

    dataset = load_iris()  # ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']
    X = dataset["data"]  # (150, 4)
    y = dataset["target"]  # (150,)

    return [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)) for (x, y) in zip(X, y)]
