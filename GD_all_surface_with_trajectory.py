from utils.GD_variations import *
from utils.Graphs import make_GD_trajectory_graphs


A = np.array([[3, 2],
              [2, 4]
              ])
b = np.array([5, 6])


def f(x: np.ndarray) -> np.ndarray:
    return x.T @ A @ x + b.T @ x


def grad_f(x: np.ndarray) -> np.ndarray:
    return (A + A.T) @ x + b


x0 = np.array([6, 7])

# Константа
Hessian_f = 2 * A
L = np.max(np.linalg.eigvalsh(Hessian_f))
x_opt_constant, history_sol_constant = GD_constant(grad_f, x0=x0, alpha=1/L)[:2]

# Адаптивный поиск константы Липшица
x_opt_adapt_Lipschitz, history_sol_adapt_Lipschitz = GD_adaptive_Lipschitz(f, grad_f, x0=x0,
                                                                                 L0=1.3, gamma=1.7, rho=0.2)[:2]

# Через одномерную оптимизацию
x_opt_ZeroOpt, history_sol_ZeroOpt = GD_and_ZeroOpt(f, grad_f, x0=x0)[:2]

# Backtracking
x_opt_backtracking, history_sol_backtracking = GD_backtracking(f, grad_f, x0=x0,
                                                               alpha0=0.4, c1=0.1, c2=0.7, rho=0.9)[:2]



x = np.arange(-7, 7, 0.1)
y = np.arange(-7, 7, 0.1)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = np.array([f(p) for p in np.stack([X.ravel(), Y.ravel()], axis=1)]).reshape(X.shape)

z_values = np.apply_along_axis(f, 1, history_sol_constant)
history_sol_constant_3d = np.column_stack([history_sol_constant, z_values])

z_values = np.apply_along_axis(f, 1, history_sol_adapt_Lipschitz)
history_sol_adapt_Lipschitz_3d = np.column_stack([history_sol_adapt_Lipschitz, z_values])

z_values = np.apply_along_axis(f, 1, history_sol_ZeroOpt)
history_sol_ZeroOpt_3d = np.column_stack([history_sol_ZeroOpt, z_values])

z_values = np.apply_along_axis(f, 1, history_sol_backtracking)
history_sol_backtracking_3d = np.column_stack([history_sol_backtracking, z_values])

data_list = [
    {'X': X, 'Y': Y, 'Z': Z, 'x0': np.array([*x0, f(x0)]),
     'x_opt': np.array([*x_opt_constant, f(x_opt_constant)]), 'history': history_sol_constant_3d},

    {'X': X, 'Y': Y, 'Z': Z, 'x0': np.array([*x0, f(x0)]),
     'x_opt': np.array([*x_opt_adapt_Lipschitz, f(x_opt_adapt_Lipschitz)]), 'history': history_sol_adapt_Lipschitz_3d},

    {'X': X, 'Y': Y, 'Z': Z, 'x0': np.array([*x0, f(x0)]),
     'x_opt': np.array([*x_opt_ZeroOpt, f(x_opt_ZeroOpt)]), 'history': history_sol_ZeroOpt_3d},

    {'X': X, 'Y': Y, 'Z': Z, 'x0': np.array([*x0, f(x0)]),
     'x_opt': np.array([*x_opt_backtracking, f(x_opt_backtracking)]), 'history': history_sol_backtracking_3d},
]

titles = [
    r'Константа $\alpha = const = \frac{1}{L}$',
    r'Адаптивный поиск $L$',
    r'Поиск $\beta$ через опт. нулевого порядка ($\alpha_k = \frac{\beta}{||\nabla f(x_k)||}$)',
    'Условия Армихо-Вульфа (backtracking)'
]

make_GD_trajectory_graphs(data_list, titles, (25, 160))