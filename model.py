from copy import deepcopy
import numpy as np
from numpy import ndarray, zeros, ones, array
import pandas as pd
from heapq import heappop, heappush, heapify
from typing import TypedDict
import random

import time

INFINITY = np.inf
NAN = np.nan

# creates an numpy array with shape as input initialized to infinity
def infinities(shape) -> ndarray:
    return np.full(shape, INFINITY)


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

class DIJKSTRA_RES(TypedDict):
    dist: dict[Node, float]
    prev: dict[Node, Node]


XFC_RES = dict[Node, DIJKSTRA_RES]
DUMMY_NODE = Node(-1)

class Graph:

    xfc_list : list[Node]

    def __init__(self, nodes : list[Node], links : set[Link], xfc_list : list[int] = []) -> None:
        self.nodes : list[Node] = nodes
        self.node_dict : dict[int, Node] = {node.id: node for node in nodes}
        self.linkset : set[Link]= links
        self.links : dict[tuple[Node, Node], Link] = {(link.start, link.end): link for link in links}
        self.assign_xfc(xfc_list)


    def assign_xfc(self, xfc_list : list[int]):
        """
        Assigns the `xfc_list` to the `XFC` attribute of each node in the graph.
        
        Parameters:
            xfc_list (list[int]): A list of node IDs to assign as `XFC` values.
        
        Returns:
            None
        """
        try:
            len(xfc_list)
        except:
            return
        
        for node in self.nodes:
            node.XFC = node.id in xfc_list
        self.xfc_list = [self.node_dict[xfc] for xfc in xfc_list]

    
    def determine_xfc(self, n : int | float, method='degree') -> list[Node]:
        if n < 1:
            # percentage
            n_nodes = int(n * len(self.nodes))
            n_nodes = max(n_nodes, 1)
        else:
            n_nodes = int(n)
        # select n random nodes from the nodeset
        if method == 'random':
            selected_nodes = random.sample(self.nodes, max(n_nodes, 1))
        elif method == 'degree':
            selected_nodes = sorted(self.nodes, key=lambda node: len(node.neighbours()), reverse=True)[:n_nodes]
        else:
            raise Exception(f'Invalid method: {method}')
        self.assign_xfc([node.id for node in selected_nodes])
        print(f'selected {n_nodes} nodes as xfc using {method=}')
        return selected_nodes
    


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
    
    def dijkstra_backward(self, origin : Node) -> tuple[dict[Node, float], dict[Node, Node]]:
        
        distances = {node: INFINITY for node in self.nodes}
        distances[origin] = 0

        prev = {}

        pq : list[tuple[float, Node]] = [(0.0, origin)]

        while pq:
            dist, node = heappop(pq)

            if dist > distances[node]:
                continue

            for link in node.inbounds:
                if distances[link.start] > distances[node] + link.BPR():
                    distances[link.start] = distances[node] + link.BPR()
                    prev[link.start] = node
                    heappush(pq, (distances[link.start], link.start))

        return distances, prev
    
    def dijkstra_for_xfc(self, xfc_list : list[Node]) -> tuple[XFC_RES, XFC_RES]:
        """
        Generate the function comment for the given function body in a markdown code block with the correct language syntax.

        Args:
            xfc_set (set[Node]): A set of XFC nodes.

        Returns:
            tuple[XFC_RES, XFC_RES]: A tuple containing the forward result and backward result.

        Raises:
            None.
        """

        def make_dict(dist, prev) -> DIJKSTRA_RES: 
            return {
                'dist': dist,
                'prev': prev,
            }

        forward_result = {}
        backward_result = {}
        for xfc in xfc_list:
            forward_result[xfc] = make_dict(*self.dijkstra(xfc))
            backward_result[xfc] = make_dict(*self.dijkstra_backward(xfc))
            
        return forward_result, backward_result

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

    def __init__(self, graph: Graph,  demands : Demands, xfc_set : list[int] | bool = []) -> None:
        self.graph = graph
        self.demands = demands
        self.graph.assign_xfc(xfc_set) # type: ignore
        self.xfc_set = self.graph.xfc_list

    
    def determine_xfc(self, n : int | float) -> None:
        self.xfc_set = self.graph.determine_xfc(n)


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
        


    def xfc_optimal(self, alpha=0.15, proning = 0) -> None:
        staged_changes = {}

        # precalculate the distance from and to XFC for all nodes to all XFC
        # print('calculating xfc distances')
        xfc_forward, xfc_backward = self.graph.dijkstra_for_xfc(self.xfc_set)


        # print(f'{xfc_forward=}\n{xfc_backward=}')
        # print('finished calculating xfc distances')
        for origin, dests in self.demands.destinations.items():

            # proning half of the xfcs based on the distance from origin to xfc
            # n_xfc = int((len(self.xfc_set)+1)/2)
            # n_xfc = max(n_xfc, 1)

            # proning to 10 xfcs that are cloest to 
            if proning:
                if proning >= 1:
                    n_xfc = min(proning, len(self.xfc_set))
                else:
                    n_xfc = max(1, int((len(self.xfc_set)+1) * proning))


                temp_xfc_list = []
                temp_xfc_list = sorted(self.xfc_set, key=lambda x: xfc_backward[x]['dist'][origin])
                temp_xfc_list = temp_xfc_list[:n_xfc]
            else:
                temp_xfc_list = self.xfc_set


            for dest in dests:

                # edge case: origin or destination is the XFC node
                if origin in self.xfc_set:
                    
                    curr = dest
                    while curr != origin:
                        if (xfc_forward[origin]['prev'][curr], curr) in staged_changes:
                            staged_changes[(xfc_forward[origin]['prev'][curr], curr)] += self.demands.dictionary[(origin, dest)] * alpha
                        else:
                            staged_changes[(xfc_forward[origin]['prev'][curr], curr)] = self.demands.dictionary[(origin, dest)] * alpha
                        curr = xfc_forward[origin]['prev'][curr] # type: ignore
                    continue
                    

                if dest in self.xfc_set:

                    curr = origin
                    while curr != dest:
                        if (curr, xfc_backward[dest]['prev'][curr]) in staged_changes:
                            staged_changes[curr, (xfc_backward[dest]['prev'][curr])] += self.demands.dictionary[(origin, dest)] * alpha
                        else:
                            staged_changes[curr, (xfc_backward[dest]['prev'][curr])] = self.demands.dictionary[(origin, dest)] * alpha
                        curr = xfc_backward[dest]['prev'][curr] # type: ignore
                    continue
                

                # calculating shortest path for the origin-destination pair
                min_dist = INFINITY
                min_xfc = DUMMY_NODE
                # for xfc in self.xfc_set:
                for xfc in temp_xfc_list:
                    # warning because the type is defined as Node | float, which causes a problem when + operator is used
                    dist = xfc_backward[xfc]['dist'][origin] + xfc_forward[xfc]['dist'][dest] # type: ignore
                    if dist < min_dist:
                        min_dist = dist
                        min_xfc = xfc

                # print(f'{min_xfc=}')
                
                # trace the min_xfc -> dest path backward and update
                curr : Node = dest
                while curr != min_xfc:
                    if (xfc_forward[min_xfc]['prev'][curr], curr) in staged_changes:
                        staged_changes[(xfc_forward[min_xfc]['prev'][curr], curr)] += self.demands.dictionary[(origin, dest)] * alpha
                    else:
                        staged_changes[(xfc_forward[min_xfc]['prev'][curr], curr)] = self.demands.dictionary[(origin, dest)] * alpha
                    curr = xfc_forward[min_xfc]['prev'][curr] # type: ignore
                
                # trace the min_xfc -> origin forward and update
                curr = origin
                while curr != min_xfc:
                    if (curr, xfc_backward[min_xfc]['prev'][curr]) in staged_changes:
                        staged_changes[(curr, xfc_backward[min_xfc]['prev'][curr])] += self.demands.dictionary[(origin, dest)] * alpha
                    else:
                        staged_changes[(curr, xfc_backward[min_xfc]['prev'][curr])] = self.demands.dictionary[(origin, dest)] * alpha
                    curr = xfc_backward[min_xfc]['prev'][curr] # type: ignore

        # print(staged_changes)  
        # print('finished assigning path')


        self.graph.discount_flow(alpha=alpha)
        self.graph.perform_change(staged_changes)


        # print('finished performing changes')



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
        


                

    def run(self, algorithm='dijkstra', alpha=0.1, threshold=0.001, maxIter = 100, method='automatic', xfc = [], verbose = False, proning = 0) -> dict[str, bool | int | float]:
        iteration_times = []
        iteration_number = 1

        def optimal_func(alpha, proning = proning) -> float:
            time_start = time.time()
            if xfc:
                self.xfc_optimal(alpha=alpha, proning=proning)
                if verbose:
                    print('running xfc optimal')
            else:
                self.optimal(alpha=alpha)
                if verbose:
                    print('running normal optimal')
            return time.time() - time_start

        iteration_times.append(optimal_func(alpha = 1.0))
        time_before = 0
        time_after = self.get_total_time()
        gap = 1
        while ((iteration_number := iteration_number + 1) <= maxIter) and abs(gap) >= threshold:
            if verbose:
                print(f'{iteration_number=}')
            iteration_times.append(optimal_func(alpha = (1/iteration_number) if method == 'automatic' else alpha))
            time_before = time_after
            time_after = self.get_total_time()
            gap = (time_before/time_after) - 1
            if verbose:
                print(f'{time_after=}, {time_before=}, {gap=}')
        if iteration_number >= maxIter:
            if verbose:
                print('max iter reached without convergence')
            return {'converge': False, 'iteration': iteration_number, 'alpha': alpha, 'time_per_iteration': sum(iteration_times)/len(iteration_times)}
        else:
            if verbose:
                print(f'converged in {iteration_number} iterations')
            return {'converge': True, 'iteration': iteration_number, 'alpha': alpha, 'time_per_iteration': sum(iteration_times)/len(iteration_times)}





    



        
