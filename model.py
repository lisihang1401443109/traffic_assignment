from copy import deepcopy
import numpy as np
from numpy import ndarray, zeros, ones, array
import pandas as pd
from heapq import heappop, heappush, heapify
from typing import TypedDict
import random
import gurobipy as grb   #type: ignore
from gurobipy import GRB #type: ignore
import networkx as nx

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
        if other is Node:
            return self.id == other.id
        else:
            return self.id == other
    
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
        self.num_nodes = len(nodes)
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

    
    def determine_xfc(self, n_nodes : int, method='degree') -> list[Node]:
        # select n random nodes from the nodeset
        if method == 'random':
            selected_nodes = random.sample(self.nodes, max(n_nodes, 1))
        elif method == 'degree':
            g = self.get_networkx_graph()
            _selected_nodes = sorted(nx.degree_centrality(g).items(), key=lambda x: x[1], reverse=True)[:n_nodes]
            selected_nodes = [self.node_dict[node[0]] for node in _selected_nodes]
        elif method == 'betweenness':
            g = self.get_networkx_graph()
            _selected_nodes = sorted(nx.centrality.betweenness_centrality(g).items(), key=lambda x: x[1], reverse=True)[:n_nodes]
            selected_nodes = [self.node_dict[node[0]] for node in _selected_nodes]
        elif method == 'eigenvector':
            g = self.get_networkx_graph()
            _selected_nodes = sorted(nx.centrality.eigenvector_centrality(g, max_iter=200, tol=1e-4).items(), key=lambda x: x[1], reverse=True)[:n_nodes]
            selected_nodes = [self.node_dict[node[0]] for node in _selected_nodes]
        elif method == 'closeness':
            g = self.get_networkx_graph()
            _selected_nodes = sorted(nx.centrality.closeness_centrality(g).items(), key=lambda x: x[1], reverse=True)[:n_nodes]
            selected_nodes = [self.node_dict[node[0]] for node in _selected_nodes]
        elif method == 'weighted_betweenness':
            g = self.get_networkx_graph()
            _selected_nodes = sorted(nx.centrality.betweenness_centrality(g, weight='length').items(), key=lambda x: x[1], reverse=True)[:n_nodes]
            selected_nodes = [self.node_dict[node[0]] for node in _selected_nodes]
        elif method == 'adjusted_degree':
            g = self.get_networkx_graph()
            _selected_nodes = []
            for i in range(n_nodes):
                # add the node that has the most degree
                _selected_nodes.append(sorted(nx.degree_centrality(g).items(), key=lambda x: x[1], reverse=True)[0][0])
                # remove the node 
                g.remove_node(_selected_nodes[-1])
            selected_nodes = [self.node_dict[node] for node in _selected_nodes]
        elif method == 'adjusted_betweenness':
            g = self.get_networkx_graph()
            _selected_nodes = []
            for i in range(n_nodes):
                print(i, '/' , n_nodes)
                _selected_nodes.append(sorted(nx.centrality.betweenness_centrality(g).items(), key=lambda x: x[1], reverse=True)[0][0])
                g.remove_node(_selected_nodes[-1])
            selected_nodes = [self.node_dict[node] for node in _selected_nodes]
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

    def get_networkx_graph(self):
        # convert graph to a networkx graph
        G = nx.DiGraph()
        for link in self.linkset:
            G.add_edge(link.start.id, link.end.id, capacity=link.capacity, flow=link.flow, length = link.fft)
        print(G.number_of_nodes(), G.number_of_edges())
        return G
        


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

    
    def determine_xfc(self, n : int | float, method='degree') -> None:
        if n < 1:
            # percentage
            n_nodes = int(n * len(self.graph.nodes))
            n_nodes = max(n_nodes, 1)
        else:
            n_nodes = int(n)
        
        # temporal
        if method == 'full_greedy':
            n_nodes = min(n_nodes, 10)
        
        if method in ['degree', 'betweenness', 'eigenvector', 'closeness', 'weighted_betweenness', 'adjusted_degree', 'adjusted_betweenness']:
            self.xfc_set = self.graph.determine_xfc(n_nodes, method)
        elif method == 'demand_in_out':
            node_demand_dict : dict[Node, float] = {node: 0.0 for node in self.graph.nodes}
            for (orig, dest), n_demand in self.demands.dictionary.items():
                node_demand_dict[orig] += n_demand
                node_demand_dict[dest] += n_demand
                
                # demand1: node_1 -> node_2, 5
                
            sorted_nodes = sorted(self.graph.nodes, key=lambda x: node_demand_dict[x], reverse=True)
            self.xfc_set = sorted_nodes[:n_nodes]
            self.graph.assign_xfc([node.id for node in self.xfc_set])
        elif method == 'demand_in_out_adj':
            node_demand_dict : dict[Node, float] = {node: 0.0 for node in self.graph.nodes}
            for (orig, dest), n_demand in self.demands.dictionary.items():
                node_demand_dict[orig] += n_demand
                node_demand_dict[dest] += n_demand
            node_demand_dict_adj : dict[Node, float] = {node: 0.0 for node in self.graph.nodes}
            for node, demand in node_demand_dict.items():
                node_demand_dict_adj[node] += demand
                for adj_link in node.neighbours():
                    adj_node = adj_link.end if adj_link.start == node else adj_link.start
                    node_demand_dict_adj[adj_node] += demand
            # self.xfc_set = [node for node in sorted(self.graph.nodes, key=lambda x: node_demand_dict_adj[x], reverse=True)][:n] #type: ignore
            sorted_nodes = sorted(self.graph.nodes, key=lambda x: node_demand_dict_adj[x], reverse=True)
            self.xfc_set = sorted_nodes[:n_nodes]
            self.graph.assign_xfc([node.id for node in self.xfc_set])
        elif method == 'greedy':
            sorted_nodes = self.xfc_init_greedy(n_nodes, mode='partial')
            self.xfc_set = sorted_nodes
            self.graph.assign_xfc([node.id for node in self.xfc_set])
        elif method == 'full_greedy':
            self.xfc_set = self.xfc_init_greedy(n_nodes, mode='full')
            self.graph.assign_xfc([node.id for node in self.xfc_set])
        else:
            print(f'undefined xfc initializing strategy \'{method}\'')
            
    def xfc_init_greedy(self, n_nodes, mode = 'partial') -> list[Node]:
        if mode == 'partial':
            node_cost = {node: 0.0 for node in self.graph.nodes}
            for node in self.graph.nodes:
                self.xfc_set = [node]
                node.XFC = True
                cost = self.run(xfc = -1)['total_cost']
                node_cost[node] = cost
                node.XFC = False
            sorted_nodes = sorted(self.graph.nodes, key=lambda x: node_cost[x], reverse=True)
            return sorted_nodes[:n_nodes]
        elif mode == 'full':
            selected_nodes = []
            for i in range(n_nodes):
                node_cost = {node: 0.0 for node in self.graph.nodes}
                for node in self.graph.nodes:
                    if node in selected_nodes:
                        continue
                    self.xfc_set = selected_nodes + [node]
                    node.XFC = True
                    cost = self.run(xfc = -1)['total_cost']
                    node_cost[node] = cost
                    node.XFC = False
                
                min_node = min(self.graph.nodes, key=lambda x: node_cost[x])
                min_node.XFC = True
                selected_nodes.append(min_node)
            return selected_nodes
        else:
            return []
        
        
    
        


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
        
    
    def longest_xfc_distance(self):
        xfc_forward, xfc_backward = self.graph.dijkstra_for_xfc(self.xfc_set)
        
        def get_closest_xfc(origin: Node) -> float:
    
            dist_list = [xfc_backward[xfc]['dist'][origin] for xfc in self.xfc_set]
            return min(dist_list)
        
        return max([
            get_closest_xfc(origin=origin) for origin in self.graph.nodes
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
        


                
    def greedy_optimal(self, alpha=0.1) -> bool:
        min_xfc = self.xfc_set[-1]
        
        staged_changes = {}
        
        xfc_forward, xfc_backward = self.graph.dijkstra_for_xfc(self.xfc_set)
        min_forward = xfc_forward[min_xfc]
        min_backward = xfc_backward[min_xfc]
        
        
        
        for origin, dest in self.demands.dictionary:        
                if origin == min_xfc:
                    
                    curr = dest
                    while curr != origin:
                        if (min_forward['prev'][curr], curr) in staged_changes:
                            staged_changes[(min_forward['prev'][curr], curr)] += self.demands.dictionary[(origin, dest)] * alpha
                        else:
                            staged_changes[(min_forward['prev'][curr], curr)] = self.demands.dictionary[(origin, dest)] * alpha
                        curr = xfc_forward[origin]['prev'][curr] # type: ignore
                    continue
                
                if dest == min_xfc:

                    curr = origin
                    while curr != dest:
                        if (curr, xfc_backward[dest]['prev'][curr]) in staged_changes:
                            staged_changes[curr, (min_backward['prev'][curr])] += self.demands.dictionary[(origin, dest)] * alpha
                        else:
                            staged_changes[curr, (min_backward['prev'][curr])] = self.demands.dictionary[(origin, dest)] * alpha
                        curr = xfc_backward[dest]['prev'][curr] # type: ignore
                    continue
                
                
                if self.demands.dictionary[(origin, dest)] == 0:
                    continue
                
                curr = dest
                while curr != min_xfc:
                    if curr not in min_forward['prev']:
                        return False
                    if (min_forward['prev'][curr], curr) in staged_changes:
                        staged_changes[(min_forward['prev'][curr], curr)] += self.demands.dictionary[(origin, dest)] * alpha
                    else:
                        staged_changes[(min_forward['prev'][curr], curr)] = self.demands.dictionary[(origin, dest)] * alpha
                    curr = min_forward['prev'][curr]
                curr = origin
                while curr != min_xfc:
                    if curr not in min_backward['prev']:
                        return False
                    if (curr, min_backward['prev'][curr]) in staged_changes:
                        staged_changes[(curr, min_backward['prev'][curr])] += self.demands.dictionary[(origin, dest)] * alpha
                    else:
                        staged_changes[(curr, min_backward['prev'][curr])] = self.demands.dictionary[(origin, dest)] * alpha
                    curr = min_backward['prev'][curr]
        
        self.graph.discount_flow(alpha=alpha)
        return True
        
        

    def run(self, algorithm='dijkstra', alpha=0.1, threshold=0.001, maxIter = 100, method='automatic', xfc = [], verbose = False, proning = 0) -> dict[str, bool | int | float]:
        iteration_times = []
        iteration_number = 1

        def optimal_func(alpha, proning = proning) -> float:
            time_start = time.time()
            if xfc > 0:
                self.xfc_optimal(alpha=alpha, proning=proning)
                if verbose:
                    print('running xfc optimal')
            elif xfc == 0:
                self.optimal(alpha=alpha)
                if verbose:
                    print('running normal optimal')
            elif xfc == -1:
                if not self.greedy_optimal(alpha=alpha):
                    return -1.0
                if verbose:
                    print('running greedy optimal')
            return time.time() - time_start

        iteration_times.append(optimal_func(alpha = 1.0))
        if iteration_times[-1] < 0:
            return {'converge': False, 'iteration': 0, 'alpha': 1.0, 'time_per_iteration': 0, 'total_cost':INFINITY, 'xfc_longest_distance': INFINITY }
        time_before = 0
        time_after = self.get_total_time()
        gap = 1
        while ((iteration_number := iteration_number + 1) <= maxIter) and abs(gap) >= threshold:
            if verbose:
                print(f'{iteration_number=}')
            iteration_times.append(optimal_func(alpha = (1/iteration_number) if method == 'automatic' else alpha))
            if iteration_times[-1] < 0:
                return {'converge': False, 'iteration': iteration_number, 'alpha': alpha, 'time_per_iteration': sum(iteration_times)/len(iteration_times), 'total_cost': INFINITY, 'xfc_longest_distance': INFINITY }
            time_before = time_after
            time_after = self.get_total_time()
            if time_after == 0.0:
                return {'converge': False, 'iteration': iteration_number, 'alpha': alpha, 'time_per_iteration': sum(iteration_times)/len(iteration_times), 'total_cost': INFINITY, 'xfc_longest_distance': INFINITY }
            gap = (time_before/time_after) - 1
            if verbose:
                print(f'{time_after=}, {time_before=}, {gap=}')
        if iteration_number >= maxIter:
            if verbose:
                print('max iter reached without convergence')
            return {'converge': False, 'iteration': iteration_number, 'alpha': alpha, 'time_per_iteration': sum(iteration_times)/len(iteration_times), 'total_cost': self.get_total_time(), 'xfc_longest_distance': self.longest_xfc_distance() }
        else:
            if verbose:
                print(f'converged in {iteration_number} iterations')
            return {'converge': True, 'iteration': iteration_number, 'alpha': alpha, 'time_per_iteration': sum(iteration_times)/len(iteration_times), 'total_cost': self.get_total_time(), 'xfc_longest_distance': self.longest_xfc_distance() }


    # def get_gp_model(self, num_xfc):
    #     m = grb.Model()
    #     # BPR: fft * (1 + alpha * pow((flow/capacity), beta))
    #     links = self.graph.linkset
        
    #     dist_dict = {}
    #     # use dijkstra to compute node-to-node distances
    #     for node in self.graph.nodes:
    #         dist, prev = self.graph.dijkstra(node)
    #         for n, d in dist.items():
    #             if d == INFINITY:
    #                 dist_dict[node, n] = GRB.INFINITY
    #                 continue
    #             dist_dict[node, n] = d
                
    #     print(dist_dict)
        
    #     # xfc variables: binary variable indicating whether a node is xfc
    #     is_xfc = m.addVars([node.id for node in self.graph.nodes], vtype=GRB.BINARY, name='is_xfc')
        
    #     # sum_constraint: the number of xfcs should not exceed num_xfc
    #     m.addConstr(is_xfc.sum() == num_xfc, name='num_xfc_constraint')
        
    #     # variables indicating minimal distance from node to xfc
    #     min_dist = m.addVars([node.id for node in self.graph.nodes], vtype=GRB.CONTINUOUS, name='min_dist')
    #     xfc_min_dist = m.addVars([node.id for node in self.graph.nodes], vtype=GRB.CONTINUOUS, name='xfc_min_dist')
        
    #     # variables indicating distance from any node to any xfc
    #     xfc_dists = m.addVars([node.id for node in self.graph.nodes], [node.id for node in self.graph.nodes], vtype=GRB.CONTINUOUS, name='xfc_dists')
    #     for i in self.graph.nodes:
    #         for j in self.graph.nodes:
    #             # print(i, j)
    #             if i == j:
    #                 m.addConstr(xfc_dists[i.id, j.id] == 0, name = f'self_dist_constraint_{i.id}_{j.id}')
    #                 continue
    #             if dist_dict[i.id, j.id] == GRB.INFINITY:
    #                 m.addConstr(xfc_dists[i.id, j.id] == GRB.INFINITY, name=f'inf_dist_constraint_{i.id}_{j.id}')
    #                 continue
    #             m.addConstr((is_xfc[j.id] == 1) >> (xfc_dists[i.id, j.id] == dist_dict[i.id, j.id]), name=f'xfc_dist_constraint_{i.id}_{j.id}')
    #             m.addConstr((is_xfc[j.id] == 0) >> (xfc_dists[i.id, j.id] == GRB.INFINITY), name=f'non_xfc_dist_constraint_{i.id}_{j.id}')
        
    #     # min_dist[a] = min_(dist[a, xfc_dists])
        
    #     for i in self.graph.nodes:
    #         m.addConstr(min_dist[i.id] == grb.min_(xfc_dists.select(i.id, '*')), name=f'min_dist_constraint_{i}')
            
    #     for i in self.graph.nodes:
    #         m.addConstr((is_xfc[i.id] == 1) >> (xfc_min_dist[i.id] == 0), name=f'xfc_min_dist_constraint_{i}')
    #         m.addConstr((is_xfc[i.id] == 0) >> (xfc_min_dist[i.id] == min_dist[i.id]), name=f'non_xfc_min_dist_constraint_{i}')
            
    #     # objective: minimize the sum of all min_dist
    #     m.setObjective(xfc_min_dist.sum(), GRB.MINIMIZE)
        
    #     return m
    
    
    def get_gp_model(self, num_xfc):
        m = grb.Model()
        
        dist_dict = {}
        MAX_DIST = 1
        # use dijkstra to compute node-to-node distances
        for node in self.graph.nodes:
            dist, prev = self.graph.dijkstra(node)
            for n, d in dist.items():
                if d == INFINITY:
                    dist_dict[node, n] = GRB.INFINITY
                    continue
                dist_dict[node, n] = d
                MAX_DIST = max(MAX_DIST, d)
        
        print(dist_dict)
        print(MAX_DIST)
        
        # xfc variables: binary variable indicating whether a node is xfc
        is_xfc = m.addVars([node.id for node in self.graph.nodes], vtype=GRB.BINARY, name='is_xfc')
        
        # sum_constraint: the number of xfcs should not exceed num_xfc
        m.addConstr(is_xfc.sum() == num_xfc, name='num_xfc_constraint')
        
        dist_var_dict = {}
        xfc_var_dict = {}
        # dist_variables: recording the distance from node to any node
        for i in self.graph.nodes:
            for j in self.graph.nodes:
                if dist_dict[i, j] == GRB.INFINITY:
                    continue
                
                dist_var_dict[(i, j)] = m.addVar(name=f'dist_{i}_{j}', vtype=GRB.CONTINUOUS, lb=0)
                xfc_var_dict[(i, j)] = m.addVar(name=f'xfc_{i}_{j}', vtype=GRB.CONTINUOUS, lb=0)
                
                m.addConstr(dist_var_dict[(i, j)] == dist_dict[i, j], name=f'dist_constraint_{i}_{j}')
                m.addConstr(xfc_var_dict[(i, j)] == is_xfc[j] * MAX_DIST + dist_var_dict[(i, j)], name=f'xfc_constraint_{i}_{j}')
        
        
        min_var_dict = {}
        # min_xfc_dist_variables: recording the distance from node to nearest xfc
        for node in self.graph.nodes:
            min_var_dict[node] = m.addVar(name=f'min_{node}', vtype=GRB.CONTINUOUS, lb=0)
            m.addConstr(min_var_dict[node] == grb.min_([
                xfc_var_dict[(node, i)] for i in self.graph.nodes if (node, i) in xfc_var_dict]), name=f'min_constraint_{node}')
        
        # objective: minimize the sum of all min_dist
        m.setObjective(grb.quicksum([min_var_dict[node] for node in self.graph.nodes]), GRB.MINIMIZE)
        
        return m
        
        
        
            

    # def get_gp_model(self):
    #     m = grb.Model()
    #     # flow_matrix = np.zeros((self.graph.num_nodes, self.graph.num_nodes))
    #     # alphas_matrix = np.zeros((self.graph.num_nodes, self.graph.num_nodes))
    #     # betas_matrix = np.zeros((self.graph.num_nodes, self.graph.num_nodes))
    #     # capacities_matrix = np.zeros((self.graph.num_nodes, self.graph.num_nodes))
    #     # fft_matrix = np.zeros((self.graph.num_nodes, self.graph.num_nodes))

    #     # BPR: fft * (1 + alpha * pow((flow/capacity), beta))
    #     links = self.graph.linkset
    #     flow_dict = {}
    #     time_dict = {}
    #     temp_dict = {}
    #     for link in links:
    #         flow_dict[link] = m.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f'flow_{link.start.id}_{link.end.id}') # type: ignore
    #         time_dict[link] = m.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f'time_{link.start.id}_{link.end.id}') # type: ignore
    #         temp_dict[link] = m.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f'temp_{link.start.id}_{link.end.id}') # type: ignore
    #         # temp = flow / capacity
    #         m.addConstr(temp_dict[link] == flow_dict[link] / link.capacity, name=f'flow_{link.start.id}_{link.end.id}')
    #         # BPR_constraints
    #         m.addConstr(link.fft * (1 + link.alpha * pow(temp_dict[link], link.beta)) == time_dict[link], name=f'BPR_{link.start.id}_{link.end.id}')
        
        

    #     # constraint for each OD pair
    #     origins = {}
    #     for origin, dests in self.demands.destinations.items():
    #         origin_dict = {}
    #         for link in links:
    #             origin_dict[link] = m.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f'origin_{origin}_link_{link.start.id}_{link.end.id}') # type: ignore
            
    #         # origin constraint: 
    #         m.addConstr(grb.quicksum(origin_dict[link] for link in links if link.start.id == origin) - grb.quicksum(origin_dict[link] for link in links if link.end.id == origin) == sum(self.demands.destinations[origin]), name=f'origin_{origin}')

    #         for dest in dests:
    #             # destination constraint
    #             m.addConstr(grb.quicksum(origin_dict[link] for link in links if link.start.id == dest) - grb.quicksum(origin_dict[link] for link in links if link.end.id == dest) == sum(self.demands.destinations[dest]), name=f'origin_{origin}_destination_{dest}')

    #         origins[origin] = origin_dict

    #     # flows from each origin should add up to totol flow for a link
    #     for link in links:
    #         m.addConstr(grb.quicksum(
    #             origins[origin][link] for origin in self.demands.destinations
    #         ) == flow_dict[link], name=f'sum_flow_{link.start.id}_{link.end.id}')

    #     # objective: sum of time
    #     m.setObjective(
    #         grb.quicksum(
    #             time_dict[link] for link in links
    #         ), GRB.MINIMIZE
    #     )

    #     m.update()
    #     m.optimize()
        

            
        
                



    



        
