import scipy.sparse as sp


def to_dense_matrix(X):
    """Compatibility function for serialized AutoML Forge pipelines."""
    if sp.issparse(X):
        return X.toarray()
    return X
