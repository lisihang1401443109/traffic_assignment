import os
import pandas as pd
from model import *


def load_from_folder(folder = './inputs/SiouxFalls/'):
    input_folder = folder

    net = input_folder + [i  for i in os.listdir(input_folder) if 'net' in i][0]
    trips = input_folder + [i for i in os.listdir(input_folder) if 'trips' in i][0]

    net_df = pd.read_csv(net, sep='\t')
    trips_df = pd.read_csv(trips, sep='\t')

    return net_df, trips_df

def create_graph_and_demands_from_inputs(net_df, trips_df, xfc = []):
    node_dict = dict()
    link_set = set()
    for ind in net_df.index:
        link = net_df.iloc[ind]
        if link.init_node not in node_dict:
            node_dict[link.init_node] = Node(int(link.init_node))
        if link.term_node not in node_dict:
            node_dict[link.term_node] = Node(int(link.term_node))
        start_node = node_dict[link.init_node]
        end_node = node_dict[link.term_node]

        n_link = Link(start=start_node, end=end_node, fft = link.free_flow_time, flow = 0, capacity = link.capacity, alpha=link.b, beta=link.power)
        link_set.add(n_link)
    

    G = Graph(nodes=list(node_dict.values()), links=link_set, xfc_list=xfc)
    print(f'{len(G.nodes)} nodes, {len(G.links)} links, ')
    D = Demands()

    for ind in trips_df.index:
        demand = trips_df.iloc[ind]
        inode = node_dict[demand.init_node]
        tnode = node_dict[demand.term_node]
        D.add_od_pair(inode, tnode, demand.demand)
    print(f'{len(D)} OD pairs')
    return G, D


def clear_outputs(output_dir):
    for folder in os.listdir(output_dir):
        if os.path.isfile(output_dir + folder):
            continue
        for file in os.listdir(output_dir + folder):
            os.remove(output_dir + folder + '/' + file)


