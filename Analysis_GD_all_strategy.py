from utils.Graphs import make_convergence_comparison_graph, make_GD_graph
from utils.GD_variations import *


A = np.array([[3, 2],
              [2, 4]
              ])
b = np.array([5, 6])


def f(x: np.ndarray) -> np.ndarray:
    return x.T @ A @ x + b.T @ x


def grad_f(x: np.ndarray) -> np.ndarray:
    return (A + A.T) @ x + b


x0 = np.array([6, 7])
f_true_opt = f(-np.linalg.solve(A + A.T, b))

# Константа
Hessian_f = 2 * A
L = np.max(np.linalg.eigvalsh(Hessian_f))
x_opt_constant, history_sol_constant = GD_constant(grad_f, x0=x0, alpha=1/L)[:2]

# Адаптивный поиск константы Липшица
x_opt_adaptive_Lipschitz, history_sol_adaptive_Lipschitz = GD_adaptive_Lipschitz(f, grad_f, x0=x0,
                                                                                 L0=1.3, gamma=1.7, rho=0.2)[:2]

# Через одномерную оптимизацию
x_opt_ZeroOpt, history_sol_ZeroOpt = GD_and_ZeroOpt(f, grad_f, x0=x0)[:2]

# Backtracking
x_opt_backtracking, history_sol_backtracking = GD_backtracking(f, grad_f, x0=x0,
                                                               alpha0=0.4, c1=0.1, c2=0.7, rho=0.9)[:2]


error_values_constant = np.array([f(x) for x in history_sol_constant]) - f_true_opt
error_values_adaptive_Lipschitz = np.array([f(x) for x in history_sol_adaptive_Lipschitz]) - f_true_opt
error_values_ZeroOpt = np.array([f(x) for x in history_sol_ZeroOpt]) - f_true_opt
error_values_backtracking = np.array([f(x) for x in history_sol_backtracking]) - f_true_opt


params = [
    (range(len(error_values_constant)), error_values_constant, r'$\alpha = const = \frac{1}{L}$'),
    (range(len(error_values_adaptive_Lipschitz)), error_values_adaptive_Lipschitz, r'Адаптивный поиск $L$'),
    (range(len(error_values_ZeroOpt)), error_values_ZeroOpt, r'Поиск $\beta$ через опт. нулевого порядка ($\alpha_k = \frac{\beta}{||\nabla f(x_k)||}$)'),
    (range(len(error_values_backtracking)), error_values_backtracking, 'Условия Армихо-Вульфа (backtracking)'),
]

make_convergence_comparison_graph(params)