import matplotlib.pyplot as plt
import numpy as np


def make_3Dgraph(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45)

    # function
    ax.plot_surface(X, Y, Z, alpha=0.8, zorder=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(r'Функция $f(x)$', pad=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def make_GD_graph(X, Y, Z, x0, x_opt, history):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45)

    # function
    ax.plot_surface(X, Y, Z, alpha=0.8, zorder=1)

    # start pos
    ax.scatter(*x0, color='purple', s=150, edgecolors='black', linewidths=1.5, zorder=10, label='Старт')

    # end pos
    ax.scatter(*x_opt, color='red', s=150, edgecolors='black', linewidths=1.5, zorder=10, label='Оптимум')

    # way
    ax.plot(history[:, 0], history[:, 1], history[:, 2], linewidth=2, label='Траектория', zorder=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Градиентный спуск', pad=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def make_convergence_comparison_graph(params):
    plt.figure(figsize=(10, 6))

    for iterations, error_values, label in params:
        plt.plot(iterations, error_values, label=label, linewidth=2)

    plt.xlabel('Итерация k')
    plt.ylabel(r'$f(x_k) - f_{opt}$')
    plt.title('Сравнение скорости сходимости (Log пространство)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()