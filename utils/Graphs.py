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


# def make_GD_trajectory_graph(X, Y, Z, x0, x_opt, history):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.view_init(elev=30, azim=45)
#
#     # function
#     ax.plot_surface(X, Y, Z, alpha=0.8, zorder=1)
#
#     # start pos
#     ax.scatter(*x0, color='purple', s=150, edgecolors='black', linewidths=1.5, zorder=10, label='Старт')
#
#     # end pos
#     ax.scatter(*x_opt, color='red', s=150, edgecolors='black', linewidths=1.5, zorder=10, label='Оптимум')
#
#     # way
#     ax.plot(history[:, 0], history[:, 1], history[:, 2], linewidth=2, label='Траектория', zorder=5)
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Градиентный спуск', pad=20, fontweight='bold')
#     ax.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


def make_GD_trajectory_graphs(data_list, titles, angles):
    assert len(data_list) == 4, "Должно быть ровно 4 набора данных"

    fig = plt.figure(figsize=(12, 7))

    for i, data in enumerate(data_list):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.view_init(elev=angles[0], azim=angles[1])

        # Surface
        ax.plot_surface(data['X'], data['Y'], data['Z'],
                        alpha=0.8, cmap='viridis', zorder=1)

        # Start pos
        ax.scatter(*data['x0'], color='purple', s=150,
                   edgecolors='black', linewidths=1.5,
                   zorder=10, label='Старт')

        # End pos
        ax.scatter(*data['x_opt'], color='red', s=150,
                   edgecolors='black', linewidths=1.5,
                   zorder=10, label='Оптимум')

        # Trajectory
        ax.plot(data['history'][:, 0], data['history'][:, 1],
                data['history'][:, 2],
                linewidth=2, color='orange',
                label='Траектория', zorder=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(titles[i], pad=20, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend()

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