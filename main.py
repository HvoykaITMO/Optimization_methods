from GD_variations import *
from Graphs import make_3Dgraph_GD



def f(x, y):
    return 2 * x**2 + 3 * y**2 + x * y


def grad_f(x, y):
    return np.array([4 * x + y, 2 * y + x])


x = np.arange(-5, 5, 0.01)
y = np.arange(-5, 5, 0.01)
X, Y = np.meshgrid(x, y, indexing='ij')

x0 = np.array([4, 4])
# x_opt, history = GD_constant(grad_f, x0=x0, alpha=0.01)
x_opt, history = GD_adaptive_Lipschitz(f, grad_f, x0=x0, L0=3, gamma=1.5, rho=0.9)
# x_opt, history = GD_and_ZeroOpt(f, grad_f, x0=x0)
print(f"Начальная точка: {x0}")
print(f"Оптимум: {x_opt}")
print(f"Траектория:\n{history}")
print(f"Количество итераций: {len(history) - 1}")


make_3Dgraph_GD(f, X, Y, x0, x_opt, history)