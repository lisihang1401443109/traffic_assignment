from traffic_types import *
import numpy as np
from numpy import ndarray, zeros, ones, array

from functools import wraps

INFINITY = np.inf
NAN = np.nan

# creates an numpy array with shape as input initialized to infinity
def infinities(shape) -> ndarray:
    return np.full(shape, INFINITY)

# being worked on
# caches the result of function with same arguments
def use_cache(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        if args in self._cache:
            return self._cache[args]
        result = func(self, *args, **kwargs)
        self._cache[args] = result
        return result
    return inner

# when called, flush the cache
def invalidator(func):
    @wraps(func)
    def inner(self, *args, **kwargs):
        self._cache = {}
        return func(self, *args, **kwargs)
    return inner


def get_flow_sum(mat: ndarray) -> float:
    sum = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i,j] in [INFINITY, np.nan]:
                continue
            sum += mat[i,j]
    return sum

class Link:

    # self.start: start Node
    # self.end: end Node
    # self.fft: free flow time of the link
    # self.flow: the flow
    # self.capacity: the capacity of the link
    # self.BPR(): the travel time of the link
    # self.__hash__(): the link with same start and end is hashed to the same
    # self.__eq__(): the links are deemed equal if with same start and end

    def __init__(self, start : Node, end : Node, fft : int = 0, flow : float = 0, capacity : float = 3.0) -> None:
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

    # self.__init__()
    #   can either initialize using list[Link] or list[Node]
    #   either initializing mode sets the other attributes
    #   note that the time and flow attributes are not initialized under node initialization
    # self.links: list[Link]
    # self.nodes: list[Node]


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
        self.calculate_time_matrix();
        
    def _initialize(self) -> None:
        self.lookup = dict([(node, self.nodes.index(node)) for node in self.nodes])
        self.fft_matrix = infinities((len(self.nodes), len(self.nodes)))
        self.flow_matrix = zeros((len(self.nodes), len(self.nodes)))
        self.capacity_matrix = ones((len(self.nodes), len(self.nodes)))


    @invalidator
    def calculate_time_matrix(self, alpha= 0.15, beta = 4):
        # in order to increase performance, we do not update the time matrix everytime change happens
        # rather we update it explicitly when we need to use the time matrix
        self.time_matrix = self.fft_matrix * (1 + alpha * np.power((self.flow_matrix / self.capacity_matrix), beta))
        print('updated time matrix:\n', self.time_matrix)


        
    @invalidator
    def _update(self, links: set[Link], alpha= 0.15, beta = 4) -> None:
        for link in links:
            self.fft_matrix[self.lookup[link.start], self.lookup[link.end]] = link.fft
            self.flow_matrix[self.lookup[link.start], self.lookup[link.end]] = link.flow
            self.capacity_matrix[self.lookup[link.start], self.lookup[link.end]] = link.capacity
            if link in self.links:
                self.links.remove(link)
            self.links.add(link)

    @invalidator
    def _assign_flow(self, new_flow_matrix: ndarray) -> None:
        self.flow_matrix = new_flow_matrix
            
    def neighbours(self, node:Node) -> list[Link]:
        return [link for link in self.links if link.start == node]

            
    @use_cache
    def shortest_path(self, origin : Node, destination : Node) -> Path:

        paths = self.dijkstra(self.lookup[origin])
        return paths[self.lookup[destination]]
    
    def dijkstra(self, origin : Node):
        '''
            being worked on
        '''
        # initialize distance and previous array
        distances = infinities(len(self.nodes))
        previous = array([None] * len(self.nodes))
        distances[self.lookup[origin]] = 0

        # worklist: while there's more nodes that has not been visited
        to_visit = set([self.lookup[node] for node in self.nodes])
        while to_visit:
            # min_index: the closest next node
            min_index = min(to_visit, key = lambda x: distances[x])
            to_visit.remove(min_index)
            neighbours = self.neighbours(self.nodes[min_index])
            # update the neighbours
            for b in neighbours:
                n = b.end
                temp_dist = self.time_matrix[min_index, self.lookup[n]] + distances[min_index]
                if distances[self.lookup[n]] > temp_dist:
                    distances[self.lookup[n]] = temp_dist
                    previous[self.lookup[n]] = min_index
        
        return previous, distances





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

    def get_sum(self):
        ret = 0.0;
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if self.matrix[i, j] in [INFINITY, NAN]:
                    continue
                ret += self.matrix[i, j]
        return ret



class Problem:


    def __init__(self, graph: Graph,  demands : Demands) -> None:
        self.graph = graph
        self.demands = demands

    def optimal(self) -> ndarray:
        # print('optimal============================================================')
        # print(f'{self.graph.lookup=}')
        new_mat = zeros((self.graph.nodes.__len__(), self.graph.nodes.__len__()))
        for i in range(self.demands.matrix.shape[0]):
            for j in range(self.demands.matrix.shape[1]):
                
                origin = self.graph.nodes[i]
                destination = self.graph.nodes[j]
                # print(origin, destination, '\n================================\n')
                amount = self.demands.matrix[i, j]
                # print(f'od:{i=}{j=}{amount=} ============================================')
                if origin == destination:
                    continue
                prev, dist = self.graph.dijkstra(origin)
                cnode = self.graph.lookup[destination]
                # print(prev)
                while not cnode == self.graph.lookup[origin]:
                    if cnode is None:
                        break
                    # print(f'{prev[cnode]= }, {cnode=}')
                    new_mat[prev[cnode], cnode] += amount
                    cnode = prev[cnode]
                # print(f'endod:{i=}{j=}{amount=} ============================================')
        # print('optimal_end=========================================================')
        
        return new_mat


    def get_total_time(self):
        # we define total time the simple some of all the individual's travel time
        # to do this, for sum(num_in_link * link_travel_time)
        time_matrix_cpy = self.graph.time_matrix.copy()
        time_matrix_cpy[time_matrix_cpy == INFINITY] = 0
        return sum(np.reshape(time_matrix_cpy * self.graph.flow_matrix, (-1, )))
        
        

    def run(self, algorithm='dijkstra', alpha=0.15, threshold=0.05):
        print('initializing============================================================')
        self.graph.calculate_time_matrix();
        optimal = self.optimal()
        self.graph._assign_flow(optimal)
        self.graph.calculate_time_matrix();
        i=0
        print(f'iteration{(i := i+1)=}================================================================')
        old_time = self.get_total_time()
        new_time = -1
        while new_time == -1 or np.absolute(old_time - new_time) >= threshold:
            opt_mat = self.optimal()
            # print(opt_mat)
            # print('================================================================')
            # print(self.graph.flow_matrix)
            self.graph._assign_flow(self.graph.flow_matrix * (1-alpha) + opt_mat * alpha)
            # print('================================================================')
            # print(self.graph.flow_matrix)
            # assert(get_flow_sum(self.graph.flow_matrix) == demand_sum)
            old_time = new_time if not new_time == -1 else old_time
            new_time = self.get_total_time()
            self.graph.calculate_time_matrix();
            print(f'{new_time=}, \n{old_time=} \n================================================')



    



        
