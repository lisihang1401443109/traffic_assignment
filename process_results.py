import os
from matplotlib import pyplot as plt
import json
from cycler import cycler


MARKERS = ['.','o','v','^','<','>','1','2','3','4','s','p']
LINES = [
    (0, ()),         # solid
    (0, (1, 1)),     # densely dashed   
    (0, (5, 1)),     # loosely dashed
    (0, (3, 1, 1, 1)), # densely dashed with dots
    (0, (3, 1, 1, 1, 1, 1)), # loosely dashed with dots
    (0, (5, 10)),    # loosely dashed 
    (0, (1, 10)),    # densely dashed
    (0, (3, 10, 1, 10)), # loosely dashed with dots
    (0, (3, 1, 1, 1, 1, 1, 1, 1)) # densely dashed with dots
]


plt.rcParams.update({
    'lines.color' : 'black',
})



def graph_exe_results_vary_precision(file_name = 'vary_precision.json'):
    os.chdir('/WAVE/users/unix/sli13/workspace/traffic_assignment/outputs')
    
    all_data = {}
    for network in os.listdir():
        if os.path.isdir(network):
            target_file = os.getcwd() + '/' + network + '/' + file_name
            with open(target_file, 'r') as f:
                all_data[network] = json.load(f)
                
    lines_iter = iter(LINES)            
    
    plt.figure(figsize=(20,15))
    for network, data in all_data.items():
        keys = list(data.keys())
        values = list([data[key]['total_cost'] for key in keys])
        print(values)
        one = values[0]
        values = [values/one for values in values]
        
        plt.plot(keys, values, label = network, linestyle = next(lines_iter))
    
    plt.legend()
    plt.show()
    plt.xlabel('Precision')
    plt.ylabel('Normalized Cost')
    plt.savefig('../graphs/graph_exe_results_vary_precision.png', dpi=200)
    
    
def graph_exe_results_vary_alpha(file_name = 'vary_alpha.json'):
    os.chdir('/WAVE/users/unix/sli13/workspace/traffic_assignment/outputs')
    
    all_data = {}
    for network in os.listdir():
        if os.path.isdir(network):
            target_file = os.getcwd() + '/' + network + '/' + file_name
            with open(target_file, 'r') as f:
                all_data[network] = json.load(f)
                
    lines_iter = iter(LINES)
    
    plt.figure(figsize=(20,15))
    for network, data in all_data.items():
        keys = list(data.keys())
        values = list([data[key]['total_cost'] for key in keys])
        print(values)
        one = values[0]
        values = [values/one for values in values]
        
        plt.plot(keys, values, label = network, linestyle = next(lines_iter))
    
    plt.legend()
    plt.show()
    plt.xlabel('Alpha')
    plt.ylabel('Normalized Cost')
    plt.savefig('../graphs/graph_exe_results_vary_alpha.png', dpi=200)


# def graph_exe_results_vary_centralities(file_name = 'xfc_locations_vary_centralities_0.05_0.json'):

def graph_exe_results_vary_centralities(file_name = 'xfc_centralities_1.json', xlabel = 'Centralities', ylabel = 'Normalized Cost', title = 'Normalized Cost of XFC alpha = 0.05 no proning', metric = 'total_cost', base = 'degree', output = 'graph_exe_results_vary_centralities.png'):
    os.chdir('/WAVE/users/unix/sli13/workspace/traffic_assignment/outputs')
    # os.chdir('/home/sihang/workspace/traffic_proj/outputs')
    # os.chdir('/home/vobbukyo/workspace/traffic_assignment-1/outputs')
    data_all = {}
    for network in os.listdir():
        if os.path.isdir(network):
            target_file = os.getcwd() + '/' + network + '/' + file_name
            with open(target_file, 'r') as f:
                data_all[network] = json.load(f)
                
    plt.figure(figsize=(20,15))
    # for centralities
    # plt.figure(figsize=(40,15))
    plt.rcParams['lines.markersize'] = 20
    # increase text size
    plt.rcParams.update({'font.size': 20})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    marker_iter = iter(MARKERS)
    # print(data_all)
    for network, data in data_all.items():
        try:
            _data = {}
            one = data[base][metric]
            for xfc, value in data.items():
                # data[xfc] = value/one
                _data[xfc] = one/value[metric]
                # print(xfc, one, value['total_cost'], data[xfc])
            # sorted_keys = sorted(data.keys(), key=lambda x : float(x))
            plt.scatter(_data.keys(), [_data[key] for key in _data.keys()], label = network, marker=next(marker_iter)) # type: ignore
        except:
            print('exception')
            pass
    avg = dict()
    for network, data in data_all.items():
        # calculate average total_cost
        for xfc, value in data.items():
            print(network, xfc, value, '------')
            try:
                if xfc not in avg:
                    avg[xfc] = value[metric]
                else:   
                    avg[xfc] += value[metric]
            except:
                print('exception')
    for xfc in avg:
        avg[xfc] = avg[xfc]/len(data_all)
        
    one = avg[base]
    for xfc, value in avg.items():
        avg[xfc] = one/value
    plt.plot(data.keys(), [avg[key] for key in data.keys()], label = 'average') # type: ignore
    # plt.legend()
    plt.show()
    plt.legend()
    plt.savefig(f'../graphs/{output}', dpi=200)


if __name__ == '__main__':
    # graph_exe_results_vary_precision()
    # graph_exe_results_vary_alpha()
    # graph_exe_results_vary_centralities(file_name='vary_alpha.json',
    #                                     xlabel='Alpha',
    #                                     ylabel='Normalized Cost',
    #                                     base= '0.5',
    #                                     metric = 'total_cost',
    #                                     title = 'Normalized Cost vary alpha',
    #                                     output='vary_alpha_total_cost.png',)


    graph_exe_results_vary_centralities(file_name='vary_precision.json',
                                        xlabel='Precision',
                                        ylabel='Normalized Cost',
                                        base= '0.1',
                                        metric = 'total_cost',
                                        title = 'Normalized Cost vary precision',
                                        output='vary_precision_total_cost.png',)



    # graph_exe_results_vary_centralities(file_name='xfc_results_vary_proning_0.1_degree.json',
    #                                     xlabel = 'Proning',
    #                                     ylabel = 'Cost Speed Up',
    #                                     base= '0',
    #                                     metric = 'total_cost',
    #                                     title = 'Normalized Cost of XFC precision = 0.05 vary proning',
    #                                     output = 'vary_proning_0.1_degree_total_cost.png',
    #                                     )
    
    # graph_exe_results_vary_centralities(file_name='xfc_results_vary_proning_0.1_degree.json',
    #                                     xlabel = 'Proning',
    #                                     ylabel = 'Execution Time Speed UP',
    #                                     base= '0',
    #                                     metric = 'time_per_iteration',
    #                                     title = 'Normalized Execution time of XFC precision = 0.05 vary proning',
    #                                     output = 'vary_proning_0.1_degree_time_per_iteration.png',)
    
    # graph_exe_results_vary_centralities(file_name='xfc_result_vary_centralities_0.05_0.json',
    #                                     xlabel='Centrality',
    #                                     ylabel='Cost Speed Up',
    #                                     base= 'degree',
    #                                     metric = 'total_cost',
    #                                     title = 'Normalized Cost of XFC precision = 0.05 vary centralities',
    #                                     output = 'vary_centrality_0.05_0_total_cost.png',
    #                                     )
    
    # graph_exe_results_vary_centralities(file_name='xfc_exe_time_vary_ratio.json', 
    #                              xlabel = 'xfc_ratio',
    #                              ylabel='Execution Time Speed Up',
    #                              base= '0',
    #                              metric = 'time_per_iteration',
    #                              title = 'Normalized Execution time of XFC precision = 0.05 vary xfc ratio',
    #                              output = 'vary_ratio_degree_time_per_iteration.png',
    #                              )
    
    # graph_exe_results_vary_centralities(file_name='xfc_exe_time_vary_ratio.json',
    #                                     xlabel='xfc_ratio',
    #                                     ylabel='Cost Speed Up',
    #                                     base= '0',
    #                                     metric = 'total_cost',
    #                                     title = 'Normalized Cost of XFC precision = 0.05 vary xfc ratio',
    #                                     output = 'vary_ratio_degree_total_cost.png',)