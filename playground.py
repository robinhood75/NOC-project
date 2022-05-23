import pandas as pd
from epidemics import Epidemic
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


def plot_sir():
    gamma = 0.5
    t = np.arange(0, 100, 0.01)
    y = Epidemic.get_sir_values(t, 3, 1, 1000, gamma=gamma)
    for i, name in enumerate(["S", "I", "R"]):
        plt.plot(t, y[:, i], label=name)
    plt.legend()
    plt.title(f"gamma = {gamma}")
    plt.show()


def plot(path):
    with open(path, 'r') as f:
        res = json.load(f)
    df = pd.DataFrame().from_dict(res).T.rename(columns={"n_percolation": "Avg. proportion of outbreaks",
                                                         "avg_infected": "Avg. proportion of infected cities"},
                                                index=lambda x: float(x) * 2 / 3)
    df /= 26 ** 2
    sns.lineplot(data=df)
    plt.show()


if __name__ == '__main__':
    path = r"results/results_square_lattice_26.json"
    plot(path)
