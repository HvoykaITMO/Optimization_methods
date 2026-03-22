import numpy as np
import pandas as pd

from utils.GD_variations import *
from utils.Graphs import *


pd.set_option('display.precision', 7)      # Точность чисел
pd.set_option('display.width', 120)        # Ширина вывода
pd.set_option('display.max_rows', 100)     # Макс. количество строк (20 → 100)
pd.set_option('display.max_columns', 20)   # Макс. количество столбцов
pd.set_option('display.expand_frame_repr', False)  # Не переносить строки


def f_rozenbrock(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[0], x[1]
    return (1 - x1)**2 + 100 * (x2 - x1**2)**2


def grad_f_rozenbrock(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[0], x[1]
    return np.array([-2 * (1 - x1) - 400 * (x2 - x1**2) * x1, 200 * (x2 - x1**2) * x1])


def hessian_f_rozenbrock(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[0], x[1]
    d2f_dxdx = 2 - 400*x2 + 1200*x1**2
    d2f_dxdy = -400*x1
    d2f_dydx = -400*x1
    d2f_dydy = 200
    return np.array([
        [d2f_dxdx, d2f_dxdy],
        [d2f_dydx, d2f_dydy]
    ])


# Build 3D surface graph
#-----------------------------------------------------------------------------------------------------------------------
# For 3d surface
x = np.arange(-7, 7, 0.1)
y = np.arange(-7, 7, 0.1)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = np.array([f_rozenbrock(p) for p in np.stack([X.ravel(), Y.ravel()], axis=1)]).reshape(X.shape)

# Create surface graph
make_3Dgraph(X, Y, Z)
#-----------------------------------------------------------------------------------------------------------------------


# Analysis L:
#-----------------------------------------------------------------------------------------------------------------------
#Гессиан функции Розенброка зависит от x^2 и y, поэтому его собственные значения неограниченно растут при удалении от
# начала координат. Следовательно, не существует глобальной константы Липшица для градиента.
#-----------------------------------------------------------------------------------------------------------------------


# Analysis mu:
#-----------------------------------------------------------------------------------------------------------------------
#Гессиан функции Розенброка:
#∇²f(x,y) = [[2 - 400y + 1200x²,  -400x],
#            [-400x,               200]]

#Минимальное собственное значение:
#λ_min = f(x, y) — зависит от координат
# Определитель Гессиана:
# det(∇²f) = 400 + 80000(x² - y)
#
# При y > x² + 0.005 определитель отрицателен → Гессиан не
# положительно определён → функция не выпукла в этой точке.
#-----------------------------------------------------------------------------------------------------------------------

# # Lambdas:
# # lambda1_2 = tr_Hesse +- sqrt((tr_Hesse / 2)^2 - det(Hesse))
# d2f_dxdx = lambda x: 2 - 400*x[1] + 1200*x[0]**2
# d2f_dxdy = lambda x: -400*x[0]
# d2f_dydx = lambda x: -400*x[0]
# d2f_dydy = 200
#
# tr_Hesse = lambda x: np.linalg.trace(hessian_f_rozenbrock(x))
# det_Hesse = lambda x: np.linalg.det(hessian_f_rozenbrock(x))
# lambda1 = lambda x: tr_Hesse(x)/2 + np.sqrt((tr_Hesse(x)**2)/4 - det_Hesse(x))
# lambda2 = lambda x: tr_Hesse(x)/2 - np.sqrt((tr_Hesse(x)**2)/4 - det_Hesse(x))

# GD covergence
#-----------------------------------------------------------------------------------------------------------------------
x0 = np.array([5, 5])
x_true_opt = np.array([1, 1])
f_true_opt = f_rozenbrock(x_true_opt)  # Аналитический минимум

# Константа
x_opt_constant, history_sol_constant = GD_constant(grad_f_rozenbrock, x0=x0, alpha=0.0001)[:2]

# Адаптивный поиск константы Липшица
x_opt_adapt_Lipschitz, history_sol_adapt_Lipschitz = GD_adaptive_Lipschitz(f_rozenbrock, grad_f_rozenbrock, x0=x0,
                                                                                 L0=0.001, gamma=1.7, rho=0.2)[:2]

# Через одномерную оптимизацию
x_opt_ZeroOpt, history_sol_ZeroOpt = GD_and_ZeroOpt(f_rozenbrock, grad_f_rozenbrock, x0=x0)[:2]

# Backtracking
x_opt_backtracking, history_sol_backtracking = GD_backtracking(f_rozenbrock, grad_f_rozenbrock, x0=x0,
                                                               alpha0=0.01, c1=0.2, c2=0.6, rho=0.9)[:2]


error_values_constant = np.array([f_rozenbrock(x) for x in history_sol_constant]) - f_true_opt
error_values_adaptive_Lipschitz = np.array([f_rozenbrock(x) for x in history_sol_adapt_Lipschitz]) - f_true_opt
error_values_ZeroOpt = np.array([f_rozenbrock(x) for x in history_sol_ZeroOpt]) - f_true_opt
error_values_backtracking = np.array([f_rozenbrock(x) for x in history_sol_backtracking]) - f_true_opt


params = [
    (range(len(error_values_constant)), error_values_constant, r'$\alpha = const = \frac{1}{L}$'),
    (range(len(error_values_adaptive_Lipschitz)), error_values_adaptive_Lipschitz, r'Адаптивный поиск $L$'),
    (range(len(error_values_ZeroOpt)), error_values_ZeroOpt, r'Поиск $\beta$ через опт. нулевого порядка ($\alpha_k = \frac{\beta}{||\nabla f(x_k)||}$)'),
    (range(len(error_values_backtracking)), error_values_backtracking, 'Условия Армихо-Вульфа (backtracking)'),
]

iterations = [len(history_sol_constant) - 1, len(history_sol_adapt_Lipschitz) - 1,
              len(history_sol_ZeroOpt) - 1, len(history_sol_backtracking) - 1]
approx_opts = [ np.round(x_k, 6) for x_k in [x_opt_constant, x_opt_adapt_Lipschitz, x_opt_ZeroOpt, x_opt_backtracking]]
error = [np.round(np.abs(x_true_opt - x_k), 6) for x_k in approx_opts]

df_results = pd.DataFrame({"Итераций": iterations, "Результат работы": approx_opts, "Ошибка": error})
print(df_results)

make_convergence_comparison_graph(params)

# GD methods
#-----------------------------------------------------------------------------------------------------------------------
x = np.arange(-7, 7, 0.1)
y = np.arange(-7, 7, 0.1)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = np.array([f_rozenbrock(p) for p in np.stack([X.ravel(), Y.ravel()], axis=1)]).reshape(X.shape)

z_values = np.apply_along_axis(f_rozenbrock, 1, history_sol_constant)
history_sol_constant_3d = np.column_stack([history_sol_constant, z_values])

z_values = np.apply_along_axis(f_rozenbrock, 1, history_sol_adapt_Lipschitz)
history_sol_adapt_Lipschitz_3d = np.column_stack([history_sol_adapt_Lipschitz, z_values])

z_values = np.apply_along_axis(f_rozenbrock, 1, history_sol_ZeroOpt)
history_sol_ZeroOpt_3d = np.column_stack([history_sol_ZeroOpt, z_values])

z_values = np.apply_along_axis(f_rozenbrock, 1, history_sol_backtracking)
history_sol_backtracking_3d = np.column_stack([history_sol_backtracking, z_values])

data_list = [
    {'X': X, 'Y': Y, 'Z': Z, 'x0': np.array([*x0, f_rozenbrock(x0)]),
     'x_opt': np.array([*x_opt_constant, f_rozenbrock(x_opt_constant)]), 'history': history_sol_constant_3d},

    {'X': X, 'Y': Y, 'Z': Z, 'x0': np.array([*x0, f_rozenbrock(x0)]),
     'x_opt': np.array([*x_opt_adapt_Lipschitz, f_rozenbrock(x_opt_adapt_Lipschitz)]), 'history': history_sol_adapt_Lipschitz_3d},

    {'X': X, 'Y': Y, 'Z': Z, 'x0': np.array([*x0, f_rozenbrock(x0)]),
     'x_opt': np.array([*x_opt_ZeroOpt, f_rozenbrock(x_opt_ZeroOpt)]), 'history': history_sol_ZeroOpt_3d},

    {'X': X, 'Y': Y, 'Z': Z, 'x0': np.array([*x0, f_rozenbrock(x0)]),
     'x_opt': np.array([*x_opt_backtracking, f_rozenbrock(x_opt_backtracking)]), 'history': history_sol_backtracking_3d},
]

titles = [
    r'Константа $\alpha = const = \frac{1}{L}$',
    r'Адаптивный поиск $L$',
    r'Поиск $\beta$ через опт. нулевого порядка ($\alpha_k = \frac{\beta}{||\nabla f(x_k)||}$)',
    'Условия Армихо-Вульфа (backtracking)'
]

make_GD_trajectory_graphs(data_list, titles, angles=(20, 120))

