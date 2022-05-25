import json
import os
from epidemics import Epidemic
import numpy as np
import utils
import time


def run(static_graph_,
        v_init_,
        p,
        n=1e3,
        lambda_=3,
        mu=1,
        verbose=True,
        pc_=None,
        time_limit=None,
        dt=None,
        max_t=None):
    """
    Runs the simulation until it reaches max_t or every town gets infected.

    :param static_graph_: static graph
    :param v_init_: initialize the epidemic at vertex v_init
    :param p: travel rate between towns
    :param n: population of every town
    :param lambda_:
    :param mu:
    :param verbose:
    :param pc_: percolation threshold of the graph. If specified, prints an estimate of pandemic threshold
    :param time_limit: include a time limit for the simulation. If simulation time > time_limit, stop simulation
    :param dt:
    :param max_t:

    :return: bool(all towns are infected), number of infected towns
    """
    p_star = p * (1 - mu / lambda_)
    simulation = Epidemic(static_graph=static_graph_,
                          p_star=p_star,
                          n=n,
                          lambda_=lambda_,
                          mu=mu,
                          verbose=verbose,
                          dt=dt,
                          max_t=max_t)

    if pc_ is not None:
        print(f"Predicted pandemic threshold: {utils.predict_pandemic_threshold(pc, simulation)}")

    start_time = time.time()
    timer = 0
    time_limit = np.inf if time_limit is None else time_limit
    simulation.start_simulation(v_init_)

    all_infected = False
    while simulation.time < simulation.max_t and not all_infected and timer < time_limit:
        simulation.step_simulation()
        all_infected = simulation.n_infected_towns == simulation.n_towns
        timer = time.time() - start_time
    print(f"Simulation over. Time {round(simulation.time, 4)}, {simulation.n_infected_towns} infected towns")

    return all_infected, simulation.n_infected_towns


def get_percolation_stats(static_graph_,
                          v_init_,
                          p_array_,
                          n_runs=30,
                          save_to="results.json",
                          dt=None,
                          max_t=None,
                          pc_=None,
                          verbose=False,
                          time_limit=None):
    results = {}
    print_estimate_p_th = pc_

    for p in p_array_:
        print(f"p = {p}")
        results[f"{p}"] = {
            "n_percolation": 0,
            "avg_infected": 0
        }
        d = results[f"{p}"]

        for _ in range(n_runs):
            all_infected, n_infected_towns = run(static_graph_,
                                                 v_init_,
                                                 p=p,
                                                 pc_=print_estimate_p_th,
                                                 verbose=verbose,
                                                 time_limit=time_limit,
                                                 dt=dt,
                                                 max_t=max_t)
            d["n_percolation"] += all_infected
            d["avg_infected"] += n_infected_towns / n_runs
            print_estimate_p_th = None

    with open(os.path.join("results", save_to), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    # Get static graph
    dim = 1000
    static_graph, pc = utils.get_graph(graph_name="cayley_tree", dim=dim, k=3)
    v_init = 0

    p_array = np.linspace(0.001, 0.02, 30)
    get_percolation_stats(static_graph, v_init, p_array,
                          time_limit=240, dt=0.002, max_t=5e2, n_runs=20, pc_=pc, save_to=r"cayley_tree")
