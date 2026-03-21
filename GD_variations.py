import numpy as np
from scipy.optimize import minimize_scalar


def GD_constant(grad_f, x0, alpha, epsilon=0.001, max_iterations=1000):
    x_current = x0.copy()
    history = [x_current.copy()]

    for i in range(max_iterations):
        grad = grad_f(*x_current)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < epsilon:
            break

        x_new = x_current - alpha * grad
        x_current = x_new
        history.append(x_current.copy())

    return x_current, np.array(history)


def GD_adaptive_Lipschitz(f, grad_f, x0, L0, gamma, rho, epsilon=0.001, max_iterations=1000):

    assert gamma > 1, "gamma должен быть > 1"
    assert 0 < rho < 1, "rho не в (0, 1)"
    assert L0 > 0, "L0 должен быть положительным"

    x_current = x0.copy()
    L_current = L0
    history = [x_current.copy()]
    for i in range(max_iterations):

        grad = grad_f(*x_current)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < epsilon:
            break

        f_current = f(*x_current)

        inner_iter = 0
        while True:
            x_new = x_current - 1/L_current * grad

            if f(*x_new) <= f_current - 1/(2 * L_current) * grad_norm**2:
                break
            else:
                L_current *= gamma  # gamma > 1

            inner_iter += 1
            if inner_iter > 100:
                raise RuntimeError(f"Не удалось подобрать L на итерации {i}")

        history.append(x_new.copy())
        x_current = x_new
        L_current *= rho  # rho < 1

    return x_current, np.array(history)


def GD_and_ZeroOpt(f, grad_f, x0, epsilon=0.001, max_iterations=1000):
    x_current = x0.copy()
    history = [x_current.copy()]

    for i in range(max_iterations):
        grad = grad_f(*x_current)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < epsilon:
            break

        f_along_beta = lambda beta: f(*(x_current - beta * (grad / grad_norm)))
        result = minimize_scalar(f_along_beta, method="brent")
        beta = result.x
        alpha = beta / grad_norm

        x_new = x_current - alpha * grad

        x_current = x_new
        history.append(x_current.copy())

    return x_current, np.array(history)


def GD_backtracking(f, grad_f, x0, alpha0, c1, rho, epsilon=0.001, max_iterations=1000):

    assert 0 < c1 < 1, "c1 не в (0, 1)"
    assert 0 < rho < 1, "rho не в (0, 1)"

    x_current = x0.copy()
    history = [x_current.copy()]
    for i in range(max_iterations):
        alpha_current = alpha0
        grad = grad_f(*x_current)
        grad_norm = np.linalg.norm(grad)
        phi0 = f(*x_current)

        if grad_norm < epsilon:
            break

        inner_iter = 0
        while True:
            phi = lambda alpha: f(*(x_current + alpha * (- grad)))

            if phi(alpha_current) <= phi0 + c1 * alpha_current * (- grad_norm**2):
                break

            alpha_current *= rho

            inner_iter += 1
            if inner_iter > 100:
                raise RuntimeError(f"Не удалось подобрать alpha на итерации {i}")

        x_new = x_current - alpha_current * grad

        x_current = x_new.copy()
        history.append(x_current.copy())

    return x_current, np.array(history)