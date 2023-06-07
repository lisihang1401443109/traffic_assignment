from traffic_types import *
import numpy as np
from numpy import ndarray, zeros, ones, array

from functools import wraps

INFINITY = np.inf

def infinities(shape) -> ndarray:
    return np.full(shape, INFINITY)

def use_cache(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        if args in self._cache:
            return self._cache[args]
        result = func(self, *args, **kwargs)
        self._cache[args] = result
        return result
    return inner

def invalidator(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        self._cache = {}
        return func(self, *args, **kwargs)
    return inner


class Link:

    def __init__(self, start : Node, end : Node, fft : int = 0, flow : float = 0, capacity : float = 1.0) -> None:
        self.start = start
        self.end = end
        self.fft = fft
        self.flow = flow
        self.capacity = capacity
    
    def BPR(self, alpha = 0.15, beta = 4) -> float:
        return self.fft * (1 + alpha * (self.flow/self.capacity)**beta)
    
    def __hash__(self):
        return hash((self.start, self.end))
    
    def __eq__(self, other):
        return self.start == other.start and self.end == other.end
    

class Path:

    flow : float
    time : float

    def __init__(self, links:list[Link] = [], nodes:list[Node] = []) -> None:
        if not links and not nodes:
            print("please provide a value")
        if links:
            self.links = links
            self.nodes = []
            for link in links:
                self.nodes.append(link.start)
            self.nodes.append(links[-1].end)
        if nodes:
            self.nodes = nodes
            self.links = []
            for start, end in zip(nodes[:-1], nodes[1:]):
                self.links.append(Link(start, end))
    
    def get_flow(self) -> float:
        return sum([link.flow for link in self.links])
    
    def get_time(self) -> float:
        return sum([link.BPR() for link in self.links])
    
    def get_links_from_path(self) -> list[Link]:
        res = []
        for link in self.links:
            # for path based algorithms
            res.append(Link(link.start, link.end, link.fft, self.flow, link.capacity))
    
    def get_links(self) -> list[Link]:
        return self.links



class Graph:

    fft_matrix : ndarray
    flow_matrix : ndarray
    capacity_matrix : ndarray
    time_matrix : ndarray
    lookup = dict[Node, int]

    # cache for dijkstra, make sure to invalidate when value updated
    _cache = {}

    
    def __init__(self, nodes : list[Node], links = set[Link]) -> None:
        self.nodes : list[Node] = nodes
        self.links : set[Link]= set(links)
        self._initialize()
        self._update(links)
        
    def _initialize(self) -> None:
        self.lookup = dict([(node, self.nodes.index(node)) for node in self.nodes])
        self.fft_matrix = infinities((len(self.nodes), len(self.nodes)))
        self.flow_matrix = zeros((len(self.nodes), len(self.nodes)))
        self.capacity_matrix = ones((len(self.nodes), len(self.nodes)))
        
    @invalidator
    def _update(self, links: list[Link], alpha= 0.15, beta = 4) -> None:
        for link in links:
            self.fft_matrix[self.lookup[link.start], self.lookup[link.end]] = link.fft
            self.flow_matrix[self.lookup[link.start], self.lookup[link.end]] = link.flow
            self.capacity_matrix[self.lookup[link.start], self.lookup[link.end]] = link.capacity
            self.time_matrix = self.fft_matrix * (1 + alpha * np.power((self.flow_matrix / self.capacity_matrix), beta))
            if link in self.links:
                self.links.remove(link)
            self.links.add(link)
            
    def neighbours(self, node:Node) -> list[Link]:
        return [link for link in self.links if link.start == node]

            
    @use_cache
    def shortest_path(self, origin : Node, destination : Node) -> Path:

        paths = self.dijkstra(self.lookup[origin])
        return paths[self.lookup[destination]]
    
    def dijkstra(self, origin : Node) -> list[Path]:
        '''
            being worked on
        '''

        visited = set()
        distances = infinities((len(self.nodes)))
        distances[origin] = 0

        # find the node with minimum distance
        while len(visited) < len(self.nodes):
            min_node = None
            for node in range(len(self.nodes)):
                if node not in visited and (min_node is None or distances[node] < distances[min_node]):
                    min_node = node
        
        # add the closest node to visited
        visited.add(min_node)

        for link in self.neighbours(self.nodes[min_node]):
            neighbour = self.lookup(link.end)
            


class Demands:

    matrix : ndarray
    lookup : dict[Node, int]

    def __init__(self, nodes: list[Node]) -> None:
        self.matrix : ndarray = zeros((len(nodes), len(nodes)))
        self.lookup = dict([(node, nodes.index(node)) for node in nodes])

    def set_matrix(self, matrix : ndarray) -> None:
        self.matrix = matrix;

    def add_od_pair(self, origin:Node, destination:Node, num:int) -> None:
        x = self.lookup[origin]
        y = self.lookup[destination]
        self.matrix[x, y] += num



class Problem:


    def __init__(self, graph: Graph,  demands : Demands) -> None:
        self.graph = graph
        self.demands = demands