import matplotlib.pyplot as plt


def make_3Dgraph_GD(f, X, Y, x0, x_opt, history):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=45)
    ax.plot_surface(X, Y, f(X, Y), alpha=0.8, zorder=1)
    ax.scatter(x0[0], x0[1], f(*x0), color='purple', s=150, edgecolors='black',
           linewidths=1.5,
           zorder=10,
           label='Старт')
    ax.scatter(x_opt[0], x_opt[1], f(*x_opt), color='red', s=150, edgecolors='black',
           linewidths=1.5,
           zorder=10,
           label='Оптимум')
    ax.plot(history[:, 0], history[:, 1], f(history[:, 0], history[:, 1]),
            linewidth=2,
            label='Траектория',
            zorder=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Градиентный спуск', pad=20, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()