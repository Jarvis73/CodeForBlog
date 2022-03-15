import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    plt.savefig("images/2021-06/figures-activations.png")


@ex.command
def plot_softmax():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.linspace(-5, 5, 100)
    x = np.stack((x, np.zeros_like(x)), axis=1)

    for alpha in [0.2, 0.5, 1, 2]:
        y = np.exp(x * alpha) / np.sum(np.exp(x * alpha), axis=1, keepdims=True)
        ax.plot(x[:, 0], y[:, 0])
        ax.text(x[65, 0], y[65, 0] - 0.03, "alpha={}".format(alpha))
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(y=1, color='k', linestyle='--', linewidth=0.5)
    ax.set_ylim([-0.1, 1.1])
    ax.set_title("Softmax(x * alpha)")

    plt.tight_layout()
    # plt.show()
    plt.savefig("images/2021-06/figures-softmax.png")


if __name__ == "__main__":
    ex.run_commandline()
