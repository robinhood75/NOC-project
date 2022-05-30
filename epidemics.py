import networkx as nx
import numpy as np
from scipy.integrate import odeint


class Epidemic:
    def __init__(self,
                 static_graph,
                 p_star,
                 n,
                 lambda_,
                 mu,
                 dt=0.01,
                 max_t=1e3,
                 gamma=0,
                 verbose=True,
                 model='SIR'):
        """
        :param static_graph: instance of nx.Graph
        :param n: population
        :param p_star: normalized travel rate
        :param lambda_: infection rate
        :param mu: recovery rate
        :param dt: discrete time basic unit. Default 0.001
        :param max_t: max simulation time
        :param gamma: optional rate
        :param model: 'SIR', 'SIS' or 'SIRS'
        """
        # Immutable attributes
        self.static_graph = static_graph
        self.p_star = p_star
        self.n = n
        self.lambda_, self.mu = lambda_, mu
        self.dt = dt
        self.max_t = max_t
        self.gamma = gamma
        self.sir_values = Epidemic.get_model_values(np.arange(0, max_t, dt), lambda_, mu, n, gamma=gamma, model=model)
        self.verbose = verbose
        self.model = model

        # Mutable attributes
        self.dynamic_graph = nx.Graph()
        self.t_infection = {v: None for v in list(static_graph.nodes)}
        self.n_infected = {v: 0 for v in list(static_graph.nodes)}
        self.n_recovered = {v: 0 for v in list(static_graph.nodes)}
        self.time = None
        self.new_infection = None
        self.frontier = None

    def start_simulation(self, v_init):
        """Add first vertex v_init as first infected town"""
        self.time = 0
        self.dynamic_graph.add_node(v_init)
        self.t_infection[v_init] = 0
        self.new_infection = True
        if self.verbose:
            print(f"Town {v_init} infected (time = {self.time})")

    def step_simulation(self):
        """
        Bring the simulation from t to t+dt:
        - check if an infected individual crossed the edge for every edge in the dynamic graph
        - update dynamic graph
        - update time
        - update numbers of infected & recovered people
        """
        # Compute cross-overs
        self.frontier = self.get_frontier()
        v_in = lambda e: e[0] if e[0] in list(self.dynamic_graph.nodes) else e[1]
        v_out = lambda e: e[1] if e[0] in list(self.dynamic_graph.nodes) else e[0]
        cross_overs = []
        for e in self.frontier:
            p_infection = max(self.p_star * self.dt * self.n_infected[v_in(e)], 0)  # take care of numerical issues
            if np.random.binomial(1, p_infection):
                cross_overs.append(e)
        self.new_infection = len(cross_overs) > 0

        # Update dynamic graph & times of infection
        for e in cross_overs:
            u, v = v_in(e), v_out(e)
            if v not in list(self.dynamic_graph.nodes):  # if a town gets infected by two != edges at the same time
                if self.verbose:
                    print(f"Town {v} infected (time = {round(self.time, 4)}, {self.n_infected_towns} towns infected)")
                self.dynamic_graph.add_node(v)
                self.t_infection[v] = self.time
            self.dynamic_graph.add_edge(u, v)

        # Update time
        self.time += self.dt

        # Update I & R in each town
        for v in list(self.dynamic_graph.nodes):
            assert self.t_infection[v] is not None
            time_idx = int(self.t_infection[v] // self.dt)
            _, i, r = self.sir_values[time_idx]
            self.n_infected[v], self.n_recovered[v] = i, r

    def get_frontier(self):
        """:return: list of edges on the frontier of the graph"""
        if self.new_infection:
            frontier = [[(u, v) for v in list(self.static_graph.adj[u]) if v not in list(self.dynamic_graph.nodes)]
                        for u in list(self.dynamic_graph.nodes)]
            ret = [e for lst in frontier for e in lst]   # flatten
        else:
            ret = self.frontier
        return ret

    @staticmethod
    def get_model_values(t_array, lambda_, mu, n, gamma=0, model='SIR'):
        """
        :return: array of size len(t_array) * 3 with S, I, R values for each time step
        """
        if model == 'SIS':
            def dydt(y, t):
                s, i = y
                tmp = lambda_ / n * s * i - mu * i
                return [- tmp, tmp]
            y0 = [n - 1, 1]

        elif model == 'SIR' or model == 'SIRS':
            def dydt(y, t):
                s, i, r = y
                tmp = lambda_ / n * s * i
                return [- tmp + gamma * r, tmp - mu * i, mu * i - gamma * r]
            y0 = [n - 1, 1, 0]

        else:
            raise ValueError(f"Unknown model {model}")

        sol = odeint(dydt, y0, t_array)

        if model == 'SIS':
            sol = np.concatenate([sol, np.zeros((sol.shape[0], 1))], axis=1)

        return sol

    @property
    def n_infected_towns(self):
        return len(list(self.dynamic_graph.nodes))

    @property
    def n_towns(self):
        return len(list(self.static_graph.nodes))
