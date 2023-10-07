import os
from matplotlib import pyplot as plt
import json

def graph_exe_time_results(file_name = 'xfc_exe_time_1.json'):
    os.chdir('/home/sihang/workspace/traffic_proj/outputs')

    data_all = {}
    for network in os.listdir():
        if os.path.isdir(network):
            target_file = os.getcwd() + '/' + network + '/' + file_name
            with open(target_file, 'r') as f:
                data_all[network] = json.load(f)

    # normalization
    for network, data in data_all.items():
        one = data['0']
        for xfc, value in data.items():
            data[xfc] = value/one
        
        sorted_keys = sorted(data.keys(), key=lambda x : float(x))
        plt.plot(sorted_keys, [data[key] for key in sorted_keys], label = network)

    print(data_all)
    plt.legend()
    plt.show()

def graph_exe_time_results_vary_proning(file_name = 'xfc_exe_time_4.json'):

    os.chdir('/home/sihang/workspace/traffic_proj/outputs')
    data_all = {}
    for network in os.listdir():
        if os.path.isdir(network):
            target_file = os.getcwd() + '/' + network + '/' + file_name
            with open(target_file, 'r') as f:
                data_all[network] = json.load(f)

    # normalization
    for network, data in data_all.items():
        one = data['0']
        for xfc, value in data.items():
            data[xfc] = value/one
        
        sorted_keys = sorted(data.keys(), key=lambda x : float(x))
        plt.plot(sorted_keys, [data[key] for key in sorted_keys], label = network)

    print(data_all)
    plt.legend()
    plt.show()

def graph_exe_results_vary_proning(file_name = 'xfc_results.json'):
    os.chdir('/home/sihang/workspace/traffic_proj/outputs')
    data_all = {}
    for network in os.listdir():
        if os.path.isdir(network):
            target_file = os.getcwd() + '/' + network + '/' + file_name
            with open(target_file, 'r') as f:
                data_all[network] = json.load(f)

    # normalization
    for network, data in data_all.items():
        one = data['0']
        for xfc, value in data.items():
            data[xfc] = value/one
        
        sorted_keys = sorted(data.keys(), key=lambda x : float(x))
        plt.plot(sorted_keys, [data[key] for key in sorted_keys], label = network)

    print(data_all)
    plt.legend()
    plt.show()
    
    
def graph_exe_resumts_vary_centralities(file_name = 'xfc_centralities_1.json'):
    os.chdir('/home/sihang/workspace/traffic_proj/outputs')
    # os.chdir('/home/vobbukyo/workspace/traffic_assignment-1/outputs')
    data_all = {}
    for network in os.listdir():
        if os.path.isdir(network):
            target_file = os.getcwd() + '/' + network + '/' + file_name
            with open(target_file, 'r') as f:
                data_all[network] = json.load(f)
                
    for network, data in data_all.items():
        try:
            one = data['degree']
            for xfc, value in data.items():
                data[xfc] = value/one
            # sorted_keys = sorted(data.keys(), key=lambda x : float(x))
            plt.plot(data.keys(), [data[key] for key in data.keys()], label = network)
        except:
            pass

    print(*list(data_all.items()), sep='\n')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # graph_exe_results_vary_proning()
    # graph_exe_time_results_vary_proning(file_name='xfc_exe_time_5.json')
    graph_exe_resumts_vary_centralities(file_name='xfc_centralities_2.json')


