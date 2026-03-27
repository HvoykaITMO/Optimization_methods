import numpy as np
from scipy.optimize import minimize_scalar


def GD_constant(grad_f, x0, alpha, epsilon=0.001, max_iterations=2000):
    x_current = x0.copy()
    history_solution = [x_current.copy()]

    history_grad = []
    history_grad_norm = []

    for i in range(max_iterations):
        grad = grad_f(x_current)
        grad_norm = np.linalg.norm(grad)
        history_grad.append(grad.copy())
        history_grad_norm.append(grad_norm)

        if grad_norm < epsilon:
            break

        x_new = x_current - alpha * grad
        x_current = x_new
        history_solution.append(x_current.copy())

    return x_current, np.array(history_solution), np.array(history_grad), np.array(history_grad_norm)


def GD_adaptive_Lipschitz(f, grad_f, x0, L0, gamma, rho, epsilon=0.001, max_iterations=2000):

    assert gamma > 1, "gamma должен быть > 1"
    assert 0 < rho < 1, "rho не в (0, 1)"
    assert L0 > 0, "L0 должен быть положительным"

    x_current = x0.copy()
    L_current = L0
    history_solution = [x_current.copy()]

    history_grad = []
    history_grad_norm = []

    for i in range(max_iterations):

        grad = grad_f(x_current)
        grad_norm = np.linalg.norm(grad)
        history_grad.append(grad.copy())
        history_grad_norm.append(grad_norm)

        if grad_norm < epsilon:
            break

        f_current = f(x_current)

        inner_iter = 0
        while True:
            x_new = x_current - 1/L_current * grad

            if f(x_new) <= f_current - 1/(2 * L_current) * grad_norm**2:
                break
            else:
                L_current *= gamma  # gamma > 1

            inner_iter += 1
            if inner_iter > 100:
                raise RuntimeError(f"Не удалось подобрать L на итерации {i}")

        history_solution.append(x_new.copy())
        x_current = x_new
        L_current *= rho  # rho < 1

    return x_current, np.array(history_solution), np.array(history_grad), np.array(history_grad_norm)


def GD_and_ZeroOpt(f, grad_f, x0, epsilon=0.001, max_iterations=2000):
    x_current = x0.copy()
    history_solution = [x_current.copy()]

    history_grad = []
    history_grad_norm = []

    for i in range(max_iterations):
        grad = grad_f(x_current)
        grad_norm = np.linalg.norm(grad)
        history_grad.append(grad.copy())
        history_grad_norm.append(grad_norm)

        if grad_norm < epsilon:
            break

        f_along_beta = lambda beta: f(x_current - beta * (grad / grad_norm))
        result = minimize_scalar(f_along_beta, method="brent")
        beta = result.x
        alpha = beta / grad_norm

        x_new = x_current - alpha * grad

        x_current = x_new
        history_solution.append(x_current.copy())

    return x_current, np.array(history_solution), np.array(history_grad), np.array(history_grad_norm)


def GD_backtracking(f, grad_f, x0, alpha0, c1, c2, rho, epsilon=0.001, max_iterations=2000):

    assert 0 < c2 < 1, "c2 не в (0, 1)"
    assert 0 < c1 < c2, "c1 не в (0, 1)"
    assert 0 < rho < 1, "rho не в (0, 1)"

    x_current = x0.copy()
    history_solution = [x_current.copy()]

    history_grad = []
    history_grad_norm = []

    for i in range(max_iterations):
        alpha_current = alpha0
        grad = grad_f(x_current)
        grad_norm = np.linalg.norm(grad)
        history_grad.append(grad.copy())
        history_grad_norm.append(grad_norm)

        if grad_norm < epsilon:
            break

        inner_iter = 0
        phi0 = f(x_current)
        grad_dot_d = -grad_norm**2  # dot_product(grad, -grad)
        while True:
            x_new = x_current - alpha_current * grad
            phi_new = f(x_new)  # phi(a) = f(x_k + a*d)
            grad_new = grad_f(x_new)

            armijo = phi_new <= phi0 + c1 * alpha_current * grad_dot_d  # Armijo cond. (grad_dot_d = -grad_norm**2)
            wolfe = abs(grad_new.dot(-grad)) <= c2 * abs(grad_dot_d)  # Strong Wolfe cond.

            if armijo and wolfe:
                break

            alpha_current *= rho

            inner_iter += 1
            if inner_iter > 100:
                raise RuntimeError(f"Не удалось подобрать alpha на итерации {i}")

        x_current = x_new.copy()
        history_solution.append(x_current.copy())

    return x_current, np.array(history_solution), np.array(history_grad), np.array(history_grad_norm)