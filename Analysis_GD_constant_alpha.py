import pandas as pd

from utils.GD_variations import *
from utils.Graphs import make_convergence_comparison_graph

pd.set_option('display.precision', 7)      # Точность чисел
pd.set_option('display.width', 120)        # Ширина вывода
pd.set_option('display.max_rows', 100)     # Макс. количество строк (20 → 100)
pd.set_option('display.max_columns', 20)   # Макс. количество столбцов
pd.set_option('display.expand_frame_repr', False)  # Не переносить строки

A = np.array([[3, 2],
              [2, 4]
              ])
b = np.array([5, 6])


def f(x: np.ndarray) -> np.ndarray:
    return x.T @ A @ x + b.T @ x


def grad_f(x: np.ndarray) -> np.ndarray:
    return (A + A.T) @ x + b


Hessian_f = 2 * A
spectre = np.linalg.eigvalsh(Hessian_f)
L = np.max(spectre)
mu = np.min(spectre)


x0 = np.array([3, 3])
x_opt, history_sol, history_grad, history_grad_norm = GD_constant(grad_f, x0=x0, alpha=1/L)


# analytical evaluations:
f_0 = f(x0)
f_true_opt = f(-np.linalg.solve(A + A.T, b))

# Оценка квадрата нормы градиента
#-----------------------------------------------------------------------------------------------------------------------
inequation_gk = lambda k: 2 * L * (f_0 - f_true_opt) / (k + 1)
min_grad_norm_sq = [np.min(history_grad_norm[:k+1]**2) for k in range(len(history_grad_norm))]
evaluation_gk = [np.round(inequation_gk(k), 7) for k in range(len(history_sol))]

df_gk = pd.DataFrame({
    "Итерация k": range(len(history_grad)),
    "Оценка g_k": evaluation_gk,
    "g_k": min_grad_norm_sq,
    "Текущая ||grad||^2": history_grad_norm**2,
    "g_k <= оценки?": min_grad_norm_sq <= evaluation_gk
})

print("\nОценка сходимости градиента (gk = min ||grad||^2):")
print(df_gk)
#-----------------------------------------------------------------------------------------------------------------------

# Оценка разницы f_k - f_opt после k итераций
#-----------------------------------------------------------------------------------------------------------------------
f_k_vs_f_opt_inequation = lambda k: (1 - mu/L)**k * (f_0 - f_true_opt)
evaluation_f_k_vs_f_opt = [f_k_vs_f_opt_inequation(k) for k in range(len(history_sol))]
f_k_vs_f_opt = [f(history_sol[k]) - f_true_opt for k in range(len(history_sol))]

df_f_k_vs_f_opt = pd.DataFrame({"Итерация k": range(len(history_sol)),
                                "Оценка f_k - f_opt": evaluation_f_k_vs_f_opt,
                                "f_k - f_opt": f_k_vs_f_opt,
                                "f_k - f_opt <= оценка?": f_k_vs_f_opt <= evaluation_f_k_vs_f_opt})
print(f"\nОценка невязки после k итераций (Оценка для k итерации):")
print(df_f_k_vs_f_opt)
#-----------------------------------------------------------------------------------------------------------------------

# Графики скорости сходимости
x_opt1, history_sol1, history_grad1, history_grad_norm1 = GD_constant(grad_f, x0=x0, alpha=1/L)
x_opt2, history_sol2, history_grad2, history_grad_norm2 = GD_constant(grad_f, x0=x0, alpha=0.5/L)
x_opt3, history_sol3, history_grad3, history_grad_norm3 = GD_constant(grad_f, x0=x0, alpha=0.1/L)

error_values1 = np.array([f(x) for x in history_sol1]) - f_true_opt
error_values2 = np.array([f(x) for x in history_sol2]) - f_true_opt
error_values3 = np.array([f(x) for x in history_sol3]) - f_true_opt

params = [
    (range(len(error_values1)), error_values1, r'$\alpha$ = $\frac{1}{L}$'),
    (range(len(error_values2)), error_values2, r'$\alpha$ = $\frac{0.5}{L}$'),
    (range(len(error_values3)), error_values3, r'$\alpha$ = $\frac{0.1}{L}$')
]

make_convergence_comparison_graph(params)
