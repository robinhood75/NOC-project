import networkx as nx
import numpy as np
from epidemics import Epidemic


def get_graph(graph_name, dim, k=None):
    """
    :param graph_name: "square_lattice", "1d_array", "cayley_tree"
    :param dim: dim of the graph
    :param k: to specify if graph_name is "cayley_tree"
    :return: graph, v_init, percolation threshold
    """
    if graph_name == "square_lattice":
        v_init = (dim + 1) // 2
        return nx.grid_2d_graph(dim, dim), (v_init, v_init), 0.5
    elif graph_name == "1d_array":
        return nx.grid_graph(dim=dim), 0, 1
    elif graph_name == "cayley_tree":
        assert k is not None
        return nx.full_rary_tree(r=k, n=dim), 0, 1 / (k - 1)
    else:
        raise ValueError(f"Unknown graph name {graph_name}")


def predict_pandemic_threshold(pc, simulation):
    sir = Epidemic.get_model_values(t_array=np.arange(0, simulation.max_t, simulation.dt), lambda_=simulation.lambda_,
                                    mu=simulation.mu, n=simulation.n)
    r_inf = sir[-1, 2] / simulation.n
    p_th = simulation.mu * np.abs(np.log(1 - pc)) / (simulation.n * (1 - simulation.mu / simulation.lambda_) * r_inf)
    return p_th
