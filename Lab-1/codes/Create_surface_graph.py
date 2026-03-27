from utils.Graphs import *


A = np.array([[3, 2],
              [2, 4]
              ])
b = np.array([5, 6])


def f(x: np.ndarray) -> np.ndarray:
    return x.T @ A @ x + b.T @ x


# For 3d surface
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = np.array([f(p) for p in np.stack([X.ravel(), Y.ravel()], axis=1)]).reshape(X.shape)

# Create surface graph
make_3Dgraph(X, Y, Z)