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

class NetworkEnv(gym.Env):
    def __init__(self):
        super(NetworkEnv, self).__init__()
        self.paths = [
            ['San Diego Supercomputer Center', 'SEQSUINET, Rice University, Houston', 'SURANET, Georgia Tech, Atlanta', 'Jon Von Neumann Center, Princeton, NJ'],
            ['San Diego Supercomputer Center', 'BARRnet, Palo Alto', 'Merit Univ of Michigan, Ann Arbor', 'Cornell Theory Center, Ithaca NY', 'Jon Von Neumann Center, Princeton, NJ']
        ]
        
        unique_edges = set()
        for path in self.paths:
            for u, v in zip(path[:-1], path[1:]):
                unique_edges.add(frozenset([u, v]))  
        
        self.number_of_links = len(unique_edges)        
        self.graph = create_graph()  
        self.edge_stats = [EdgeStats(u, v, self.graph[u][v]['capacity']) for u, v in self.graph.edges()]
        self.action_space = spaces.Discrete(2)     
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        self.round = 0
        self.utilizations = []

    
    def step(self, action):
        request = generate_requests(1, self.graph, min_ht=10, max_ht=20, case='I')[0]
        path = self.paths[action]
        path_edges = list(zip(path[:-1], path[1:]))
#        estats = [EdgeStats(u, v, self.graph[u][v]['capacity']) for u, v in self.graph.edges()]
        self.round += 1
        terminated = (self.round == 100)

        for edge in self.edge_stats:
            edge.remove_requests()
        success, color, _ = route(self.graph, self.edge_stats, request, path_edges)
        
        if color is not None:
            reward = 2   
        else:
            reward = -1  
        
        state = self._get_state()
        self.record_utilization()
        
        done = np.all(state == 0) 
        return state, reward, terminated, terminated, {}

    def record_utilization(self):
        current_utilization = [edge.utilization() for edge in self.edge_stats]
        self.utilizations.append(current_utilization)

    def get_episode_utilization(self):
        if self.utilizations:
            return np.mean(self.utilizations, axis=0).tolist()
        return []
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for edge in self.edge_stats:
            edge.reset()
        initial_state = self._get_state()
        self.round = 0
        self.utilizations.clear()
        return initial_state,{}
    
    def _get_state(self):
        utilizations = []
        for path in self.paths:
            for u, v in zip(path[:-1], path[1:]):
                edge = next((e for e in self.edge_stats if {e.u, e.v} == {u, v}), None)
                if edge:
                    util = edge.utilization()
                    utilizations.append(util)
                else:                    
                    utilizations.append(0.0)  

        assert len(utilizations) == 7, f"Expected 7 values, but got {len(utilizations)}"
        return np.array(utilizations, dtype=np.float32)
    
    def render(self, mode='human'):
        pass  

def route(graph, edge_stats_list, req, path_edges):

    if not path_edges:
        return None, None, None  
    
    initial_edge = path_edges[0]
    if (initial_edge[0], initial_edge[1]) in graph.edges:
        initial_capacity = graph[initial_edge[0]][initial_edge[1]]['capacity']
    else:
        return None, None, None  
    available_colors = set(range(initial_capacity))
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
        return path_edges, color, None  
    else:
        return path_edges, None, None 

