import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class Request:
    def __init__(self, s, t, ht):
        self.s = s
        self.t = t
        self.ht = ht

class EdgeStats:
    def __init__(self, u, v, cap):
        self.u = u
        self.v = v
        self.cap = cap
        self.__slots = [None] * cap
        self.__hts = [0] * cap

    def add_request(self, req: Request, color: int):
        self.__slots[color] = req
        self.__hts[color] = req.ht

    def remove_requests(self):
        for i in range(self.cap):
            if self.__slots[i] is not None and self.__hts[i] <= 1:
                self.__slots[i] = None
                self.__hts[i] = 0
            elif self.__slots[i] is not None:
                self.__hts[i] -= 1

    def get_available_colors(self) -> list[int]:
        return [i for i, slot in enumerate(self.__slots) if slot is None]

    def utilization(self):
        return sum(1 for slot in self.__slots if slot is not None) / self.cap

def generate_requests(num_reqs: int, g: nx.Graph, min_ht, max_ht, case='I') -> list[Request]:
    requests = []
    src, dst = ('San Diego Supercomputer Center', 'Jon Von Neumann Center, Princeton, NJ')
    for _ in range(num_reqs):
        if case == 'I':
            s, t = src, dst
        elif case == 'II':
            s, t = np.random.choice(g.nodes(), 2, replace=False) 
        ht = np.random.randint(min_ht, max_ht)
        requests.append(Request(s, t, ht))
    return requests


def create_graph() -> nx.Graph:
    G = nx.read_gml('./nsfnet.gml')
    nx.set_edge_attributes(G, values=10, name='capacity')
    return G

def route(g: nx.Graph, estats: list[EdgeStats], req: Request) -> (list[tuple], int, list[tuple]): # type: ignore
    path = nx.shortest_path(g, source=req.s, target=req.t)
    path_edges = list(zip(path[:-1], path[1:]))
    
    initial_capacity = g[path[0]][path[1]]['capacity']
    available_colors = set(range(initial_capacity))
    
    for u, v in path_edges:
        edge_stats = next((e for e in estats if (e.u, e.v) == (u, v) or (e.u, e.v) == (v, u)), None)
        if edge_stats:
            available_colors.intersection_update(edge_stats.get_available_colors())
    
    if available_colors:
        color = min(available_colors)
        for u, v in path_edges:
            edge_stats = next((e for e in estats if (e.u, e.v) == (u, v) or (e.u, e.v) == (v, u)), None)
            if edge_stats:
                edge_stats.add_request(req, color)
        return path, color, path_edges
    else:
        return path, None, path_edges  


def simulate_scenario(g, num_rounds, min_ht, max_ht, case):
    estats = [EdgeStats(u, v, g[u][v]['capacity']) for u, v in g.edges()]
    blocked_requests = []
    link_utilization = {edge: [] for edge in g.edges()}

    for t in range(num_rounds):
        requests = generate_requests(1, g, min_ht, max_ht, case)
        for req in requests:
            path, color, path_edges = route(g, estats, req)
            if color is not None:
                print(f"Request from {req.s} to {req.t} with holding time {req.ht}:")
                print(f"  Path: {path}")
                print(f"  Color: {color}")
            else:
                print(f"Request from {req.s} to {req.t} was blocked. No available path.")
                blocked_requests.append(req)

            for edge in g.edges():
                estat = next(e for e in estats if (e.u, e.v) == edge)
                link_utilization[edge].append(estat.utilization())
                estat.remove_requests()

    avg_link_utilization = {edge: np.mean(util) for edge, util in link_utilization.items()}
    number_of_edges = len(avg_link_utilization)
    print(number_of_edges)
    print(avg_link_utilization)
    
    network_utilization = np.mean(list(avg_link_utilization.values()))

    return blocked_requests, link_utilization, network_utilization


def plot_network_utilization(scenario_utilizations, scenario_labels):
    plt.figure(figsize=(10, 6))
    
    plt.bar(scenario_labels, scenario_utilizations, color='green')
    
    plt.title('Network-wide Utilization Across Scenarios')
    plt.xlabel('Scenario')
    plt.ylabel('Network-wide Utilization (fraction of total capacity)')
    plt.ylim(0, 1)  
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    G = create_graph()
    scenario_utilizations = []
    scenario_labels = ['Case I', 'Case II']
    scenarios = {
        'Case I': (10, 20, 'I'),
        'Case II': (10, 20, 'II')
    }

    for label, (min_ht, max_ht, case) in scenarios.items():
        blocked, utilization, net_util = simulate_scenario(G, 100, min_ht, max_ht, case)
        scenario_utilizations.append(net_util)
        print(f"Network-wide utilization for {label}: {net_util:.2f}")
    
    plot_network_utilization(scenario_utilizations, scenario_labels)

