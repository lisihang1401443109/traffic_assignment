import os
import pandas as pd
import model

def load_from_folder(folder = './inputs/SiouxFalls/'):
    input_folder = folder

    net = input_folder + [i  for i in os.listdir(input_folder) if 'net' in i][0]
    trips = input_folder + [i for i in os.listdir(input_folder) if 'trips' in i][0]

    net_df = pd.read_csv(net, sep='\t')
    trips_df = pd.read_csv(trips, sep='\t')

    return net_df, trips_df

def create_graph_and_demands_from_inputs(net_df, trips_df):
    node_set = set()
    link_set = set()
    for ind in net_df.index:
        link = net_df.iloc[ind]
        node_set.add(int(link.init_node))
        node_set.add(link.term_node)
        n_link = model.Link(start=link.init_node, end=link.term_node, fft = link.free_flow_time, flow = 0, capacity = link.capacity)
        link_set.add(n_link)

    G = model.Graph(list(node_set), link_set)
    print(f'{len(G.nodes)} nodes, {len(G.links)} links, ')
    D = model.Demands(list(node_set))

    for ind in trips_df.index:
        demand = trips_df.iloc[ind]
        D.add_od_pair(demand.init_node, demand.term_node, demand.demand)
    print(f'{len(D)} OD pairs')
    return G, D




