import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
  "text.usetex": True,
})

from sacred import Experiment

ex = Experiment("Plot")


@ex.command
def plot_activations():
    fig, ax = plt.subplots(2, 3, figsize=(10, 6))
    ax = ax.flat

    act1 = nn.ReLU()
    act2 = nn.PReLU(init=0.25)
    for var in act2.parameters():
        var.requires_grad = False
    act3 = nn.LeakyReLU(negative_slope=0.1)
    act4 = nn.ELU(alpha=1)
    act5 = nn.Sigmoid()
    act6 = nn.Tanh()
    titles = ["ReLU", "PReLU (init=0.25)", "LeakyReLU (alpha=0.1)", "ELU (alpha=1)", "Sigmoid", "Tanh"]

    x = torch.linspace(-5, 5, 100)
    y1 = act1(x)
    y2 = act2(x)
    y3 = act3(x)
    y4 = act4(x)
    y5 = act5(x)
    y6 = act6(x)

    for i in range(6):
        df = pd.DataFrame({"x": x.numpy(), "y": eval("y" + str(i + 1)).numpy()})
        sns.lineplot(x='x', y='y', data=df, ax=ax[i])
        ax[i].set_title(titles[i])
        ax[i].set_ylim([-2, 2])
        ax[i].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        ax[i].axvline(x=0, color='k', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    # plt.show()
    plt.savefig("figures-activations.png")


@ex.command
def plot_softmax():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.linspace(-5, 5, 100)
    x = np.stack((x, np.zeros_like(x)), axis=1)

    for alpha in [0.2, 0.5, 1, 2]:
        y = np.exp(x * alpha) / np.sum(np.exp(x * alpha), axis=1, keepdims=True)
        ax.plot(x[:, 0], y[:, 0])
        ax.text(x[65, 0], y[65, 0] - 0.03, r"$\alpha={}$".format(alpha))
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(y=1, color='k', linestyle='--', linewidth=0.5)
    ax.set_ylim([-0.1, 1.1])
    ax.set_title(r"Softmax($x \cdot \alpha$)")

    plt.tight_layout()
    # plt.show()
    plt.savefig("figures-softmax.png")


@ex.command
def plot_polylr():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    def polylr(it, init_lr, max_iter, eta_min=0, gamma=0.9):
        factor = eta_min + (init_lr - eta_min) * (1 - (it - 1) / (max_iter - 1)) ** gamma
        return factor

    x = np.arange(1, 1001)
    for alpha in [0.2, 0.5, 1, 2]:
        y = polylr(x, init_lr=0.1, max_iter=x.shape[0], eta_min=0.01, gamma=alpha)
        ax.plot(x, y)
        ax.text(x[650], y[650] + 0.0025, r"$\gamma={}$".format(alpha))
    ax.axhline(y=0.01, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(y=0.1, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_ylim([0, 0.11])
    ax.set_title("Polynomial learning rate")

    plt.tight_layout()
    # plt.show()
    plt.savefig("figures-polylr.png")
    

@ex.command
def plot_cosinelr():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    def cosinelr(it, init_lr, T_max, eta_min=0):
        factor = eta_min + 0.5 * (init_lr - eta_min) * (1 + np.cos(((it - 1) % T_max) / (T_max - 1) * np.pi))
        return factor

    x = np.arange(1, 1001)
    for T_max in [1000, 500]:
        y = cosinelr(x, init_lr=0.1, T_max=T_max, eta_min=0.01)
        ax.plot(x, y)
        ax.text(x[650], y[650] + 0.0025, r"$T_{max}=%s$" % T_max)
    ax.axhline(y=0.01, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(y=0.1, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_ylim([0, 0.11])
    ax.set_title("Cosine learning rate")

    plt.tight_layout()
    # plt.show()
    plt.savefig("figures-cosinelr.png")


if __name__ == "__main__":
    ex.run_commandline()
