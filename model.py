from copy import deepcopy
import numpy as np
from numpy import ndarray, zeros, ones, array
import pandas as pd
from heapq import heappop, heappush, heapify

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

# random comment
def get_flow_sum(mat: ndarray) -> float:
    sum : float = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i,j] in [INFINITY, np.nan]:
                continue
            sum += mat[i,j]
    return sum


class Node:

    def __init__(self, id : int, XFC = False) -> None:
        self.id : int = id
        self.XFC = XFC
        self.inbounds = set()
        self.outbounds = set()

    def is_XFC(self) -> bool:
        return self.XFC
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id
    
    def __str__(self) -> str:
        return f'Node({self.id})'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def neighbours(self) -> set:
        return self.inbounds | self.outbounds
    
    def __lt__(self, other):
        return self.id < other.id
    
    def __le__(self, other):
        return self.id <= other.id
    
    def __gt__(self, other):
        return self.id > other.id
    
    def __ge__(self, other):
        return self.id >= other.id
    
    def __ne__(self, other):
        return self.id != other.id


class Link:

    # self.start: start Node
    # self.end: end Node
    # self.fft: free flow time of the link
    # self.flow: the flow
    # self.capacity: the capacity of the link
    # self.BPR(): the travel time of the link
    # self.__hash__(): the link with same start and end is hashed to the same
    # self.__eq__(): the links are deemed equal if with same start and end

    def __init__(self, start : Node, end : Node, fft : float = 0, flow : float = 0, capacity : float = 3.0, alpha = 0.15, beta = 4) -> None:
        self.start : Node = start 
        self.end : Node = end
        self.fft : float = fft
        self.flow : float = flow
        self.capacity : float = capacity
        start.outbounds.add(self)
        end.inbounds.add(self)
        self.alpha = alpha
        self.beta = beta
    
    def BPR(self) -> float:
        return self.fft * (1 + self.alpha * pow((self.flow/self.capacity), self.beta))
    
    def __hash__(self):
        return hash((self.start, self.end))
    
    def __eq__(self, other):
        return self.start == other.start and self.end == other.end
    
    def __str__(self) -> str:
        return f'Link({self.start}, {self.end}, {self.fft=}, {self.flow=}, {self.capacity=})'
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def contains_xfc(self):
        return self.start.is_XFC() or self.end.is_XFC()
    
    def set_flow(self, flow):
        self.flow = flow

    def add_flow(self, flow):
        self.flow += flow
    
'''
    *not used*
'''
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
    
        return res
    def get_links(self) -> list[Link]:
        return self.links

class Graph:


    xfc_list : ndarray

    # cache for dijkstra, make sure to invalidate when value updated
    _cache = {}

    
    def __init__(self, nodes : list[Node], links : set[Link]) -> None:
        self.nodes : list[Node] = nodes
        self.linkset : set[Link]= links
        self.links : dict[tuple[Node, Node], Link] = {(link.start, link.end): link for link in links}

    


    '''
        this function is supposed to return shortest path to all nodes given the origin node
    '''
    def dijkstra(self, origin : Node) -> tuple[dict[Node, float], dict[Node, Node]]:
        distances = {node: INFINITY for node in self.nodes}
        distances[origin] = 0

        prev = {}

        pq : list[tuple[float, Node]] = [(0.0, origin)]

        while pq:
            dist, node = heappop(pq)

            if dist > distances[node]:
                continue
            
            for link in node.outbounds:
                if distances[link.end] > distances[node] + link.BPR():
                    distances[link.end] = distances[node] + link.BPR()
                    prev[link.end] = node
                    heappush(pq, (distances[link.end], link.end))
        


        return distances, prev


    def dijkstra_w_xfc(self, origin : Node, xfc : set[Node]) -> tuple[dict[Node, float], dict[Node, Node]]:

        # compute the distance with regards to each xfc
        distances = {node: INFINITY for node in self.nodes}
        distances[origin] = 0

        dist_to_xfc = []

        return self.dijkstra(origin)

    def discount_flow(self, alpha = 0.05) -> None:
        for link in self.linkset:
            link.flow *= (1-alpha)

    def perform_change(self, staged_changes: dict[tuple[Node, Node], float]) -> None:
        # print('performing changes:', staged_changes)
        for (origin, destination), flow in staged_changes.items():
            self.links[(origin, destination)].flow += flow





class Demands:

    dictionary : dict

    def __init__(self) -> None:
        self.dictionary : dict = {}
        self.destinations : dict = {}

    def set_dictionary(self, dictionary: dict):
        self.dictionary = dictionary

    def add_od_pair(self, origin:Node, destination:Node, num:int) -> None:
        self.dictionary[(origin, destination)] = num
        if origin in self.destinations:
            self.destinations[origin].append(destination)
        else:
            self.destinations[origin] = [destination]
    
    def __len__(self) -> int:
        return len(self.dictionary)
    



class Problem:

    def __init__(self, graph: Graph,  demands : Demands) -> None:
        self.graph = graph
        self.demands = demands

    def optimal(self, alpha = 0.15) -> None:

        staged_changes = {}

        for origin, dests in self.demands.destinations.items():
            dist, prev = self.graph.dijkstra(origin)
            for dest in dests:
                if self.demands.dictionary[(origin, dest)] == 0:
                    continue
                curr = dest
                while curr != origin:
                    if (prev[curr], curr) in staged_changes:
                        staged_changes[(prev[curr], curr)] += self.demands.dictionary[(origin, dest)] * alpha
                    else:
                        staged_changes[(prev[curr], curr)] = self.demands.dictionary[(origin, dest)] * alpha
                    curr = prev[curr]
        
        self.graph.discount_flow(alpha=alpha)
        self.graph.perform_change(staged_changes)




    def get_total_time(self):
        # we define total time the simple some of all the individual's travel time
        return sum([
            link.flow * link.BPR() for link in self.graph.linkset
        ])



    
    def output_result(self, output_file: str = '', log = {}) -> None:
        lst = []
        for (start, end), link in self.graph.links.items():
            lst.append({
                'start': start.id,
                'end': end.id,
                'flow': link.flow,
                'travel_time': link.BPR()
            })
        df = pd.DataFrame(lst)
        df.sort_values('start', inplace=True)
        if output_file:
            df.to_csv(output_file, index=False)
            with open(output_file[:-4]+'.dat', 'w') as f:
                f.write(f'total_time: {self.get_total_time()} units\n')
                try:
                    f.write(f"{'converged' if log['converge'] else 'not converged'} within {log['iteration']} iterations\n") 
                    f.write(f"alpha={log['alpha']}\n")
                except KeyError:
                    pass
        else:
            print(df)
        


                        
        
        

    def run(self, algorithm='dijkstra', alpha=0.1, threshold=0.001, maxIter = 100, method='automatic') -> dict[str, bool | int | float]:
        iteration_number = 1
        self.optimal(alpha = 1.0)
        time_before = 0
        time_after = self.get_total_time()
        gap = 1
        while ((iteration_number := iteration_number + 1) <= maxIter) and gap >= threshold:
            print(f'{iteration_number=}')
            self.optimal(alpha = (1/iteration_number) if method == 'automatic' else alpha)
            time_before = time_after
            time_after = self.get_total_time()
            gap = abs(time_before/time_after - 1)
            print(f'{time_after=}, {time_before=}, {gap=}')
        if iteration_number >= maxIter:
            print('max iter reached without convergence')
            return {'converge': False, 'iteration': iteration_number, 'alpha': alpha}
        else:
            print(f'converged in {iteration_number} iterations')
            return {'converge': True, 'iteration': iteration_number, 'alpha': alpha}




    



        
