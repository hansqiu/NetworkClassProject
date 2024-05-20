import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

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

    def reset(self):
        self.__slots = [None] * self.cap
        self.__hts = [0] * self.cap

def generate_requests(num_reqs: int, g: nx.Graph, min_ht, max_ht, case='II') -> list[Request]:
    requests = []
    if case == 'II':
        for _ in range(num_reqs):
            s, t = np.random.choice(list(g.nodes()), 2, replace=False)  
            ht = np.random.randint(min_ht, max_ht)
            requests.append(Request(s, t, ht))
    return requests

def create_graph() -> nx.Graph:
    G = nx.read_gml('./nsfnet.gml')
    nx.set_edge_attributes(G, values=10, name='capacity')
    return G

class NetworkEnv(gym.Env):
    def __init__(self):
        super(NetworkEnv, self).__init__()
        self.graph = create_graph()
        self.edge_stats = [EdgeStats(u, v, self.graph[u][v]['capacity']) for u, v in self.graph.edges()]
        self.request = generate_requests(1, self.graph, 10, 20, case='II')[0]
        self.paths = list(nx.all_simple_paths(self.graph, source=self.request.s, target=self.request.t))
        count = []
        for path in self.paths:
            for u, v in zip(path[:-1], path[1:]):
                edge = next((e for e in self.edge_stats if {e.u, e.v} == {u, v}), None)
                i = edge.utilization() if edge else 0
                count.append(i)
        print(len(count))
        print(self.paths)
        self.number_of_links = len(count)        
        self.action_space = spaces.Discrete(len(self.paths))
        # Set observation space dynamically based on the number of paths
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.number_of_links,), dtype=np.float32)
        

    def step(self, action):
        print(f"Taking action: {action}")
        path = self.paths[action]
        print(f"Selected path: {path}")
        path_edges = list(zip(path[:-1], path[1:]))
        self.round += 1
        terminated = (self.round == 100)

        for edge in self.edge_stats:
            edge.remove_requests()

        success, color, _ = route(self.graph, self.edge_stats, self.request, path_edges)
        if color is not None:
            reward = 2
            print(f"Request successfully routed with color {color}.")
        else:
            reward = -1
            print("Request blocked, no available color.")

        state = self._get_state()
        self.record_utilization()
        done = np.all(state == 0)
        return state, reward, terminated, terminated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for edge in self.edge_stats:
            edge.reset()
        self.request = generate_requests(1, self.graph, 10, 20, case='II')[0]
        self.paths = list(nx.all_simple_paths(self.graph, source=self.request.s, target=self.request.t))
        if not self.paths:
            print("No paths found, retrying reset.")
            return self.reset(seed, options)
        print("reset")
        unique_edges = list()
        for path in self.paths:
            for i in path:
                unique_edges.append(i) 
        print(len(unique_edges))
        print(self.paths)
        self.number_of_links = len(unique_edges)        
        num_paths = len(self.paths)
        self.action_space = spaces.Discrete(num_paths)
        # Set observation space dynamically based on the number of paths
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.number_of_links,), dtype=np.float32)
        
        print(f"Updated action and observation spaces for {self.number_of_links} paths.")
        return self._get_state(), {}

    def _get_state(self):
        utilizations = []
        for path in self.paths:
            for u, v in zip(path[:-1], path[1:]):
                edge = next((e for e in self.edge_stats if {e.u, e.v} == {u, v}), None)
                util = edge.utilization() if edge else 0
                utilizations.append(util)
        print("get") 
        print(len(utilizations))       
        return np.array(utilizations, dtype=np.float32)

    def render(self, mode='human'):
        pass

def route(graph, edge_stats_list, req, path_edges):
    print("route")
    if not path_edges:
        print("No path edges to route.")
        return None, None, None
    available_colors = set(range(graph[path_edges[0][0]][path_edges[0][1]]['capacity']))
    for u, v in path_edges:
        edge_stat = next((e for e in edge_stats_list if {e.u, e.v} == {u, v}), None)
        if edge_stat:
            available_colors.intersection_update(edge_stat.get_available_colors())
    if available_colors:
        color = min(available_colors)
        for u, v in path_edges:
            edge_stat = next((e for e in edge_stats_list if {e.u, e.v} == {u, v}), None)
            if edge_stat:
                edge_stat.add_request(req, color)
        print(f"Routing successful on path {path_edges} with color {color}.")
        return path_edges, color, None
    else:
        print(f"Routing failed on path {path_edges}, no available colors.")
        return path_edges, None, None


