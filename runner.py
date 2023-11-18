from loader import *
from model import Problem
import time
import json

ALPHA = 0.15
THRESHOLD = 0.001
MAXITER = 200
XFC_RATIO = 0.1

class Experiment:
    p : Problem

    def __init__(self, input_path: str, output_path: str, algorithm = 'dijkstra', method = 'automatic', xfc = 0, alpha=0.15, threshold=0.05, maxIter = 20, verbose = True, proning = 0, network_name = '') -> None:
        G, D = create_graph_and_demands_from_inputs(*list(load_from_folder(input_path)))
        self.input_path = input_path
        self.output_path = output_path
        self.p = Problem(G, D, network_name=network_name)
        self.algorithm = algorithm
        self.method = method
        self.alpha = alpha
        self.threashold = threshold
        self.maxIter = maxIter
        self.xfc = xfc
        self.verbose = verbose
        self.proning = proning
        self.network_name = network_name

    def __str__(self) -> str:
        return f'Experiment({self.input_path=}\n{self.output_path=}\n{self.algorithm=}\n{self.method=}\n{self.alpha=}\n{self.threashold=}\n{self.maxIter=}\n{self.xfc=}\n)'

    
    def run(self):
        print(self)
        # print(self.p)
        log = self.p.run(algorithm=self.algorithm, alpha=self.alpha, threshold= self.threashold, maxIter = self.maxIter, method = self.method, xfc=self.xfc, verbose = self.verbose, proning = self.proning)
        # self.p.output_result(f'{self.output_path}{self.algorithm}_{self.method}_{self.alpha}.csv', log)
        return log

    def exe_time(self) -> float:
        # calculate execution time of self.run()
        start_time = time.time()
        self.run()
        end_time = time.time()
        return end_time - start_time


def run_all_networks(root_folder = os.getcwd(), output_alias = 'baseline') -> None:
    print('running baseline networks')
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER)
        experiment.run()
        experiment.p.output_result(f'{out_path}{output_alias}_{experiment.method}_{experiment.alpha}_{experiment.threashold}_{experiment.maxIter}.csv')

def run_vary_precisions(root_folder = os.getcwd(), precisions = [0.1, 0.01, 0.001, 0.0001], out_alias = 'vary_precision.json') -> None:
    print(f'running vary precisions with alpha = {ALPHA}, maxIter = {MAXITER}, precisions = {precisions}')
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        results = {}
        for precision in precisions:
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=precision, maxIter = MAXITER)
            log = experiment.run()
            results[precision] = log
        with open(out_path + out_alias, 'w+') as f:
            json.dump(results, f)
            
        


def run_vary_alpha(root_folder = os.getcwd(), alphas = [0.5, 0.4, 0.3, 0.2, 0.1], out_alias = 'vary_alpha.json') -> None:
    print(f'running vary alpha with precision = {THRESHOLD}, maxIter = {MAXITER}, alpha = {alphas}')
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        results = {}
        for alpha in alphas:
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'constant', alpha=alpha, threshold=THRESHOLD, maxIter = MAXITER)
            # results.append(experiment.run())
            results[alpha] = experiment.run()
        
        with open(out_path + out_alias, 'w+') as f:
            json.dump(results, f)


def run_all_xfc(root_folder = os.getcwd(), alpha=ALPHA, xfc_ratio = XFC_RATIO, threshold = THRESHOLD) -> None:
    print(f'running all xfc with alpha = {alpha}, threshold = {threshold}, maxIter = {MAXITER}, xfc_ratio = {xfc_ratio}')
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=alpha, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False)
        experiment.p.determine_xfc(xfc_ratio)
        experiment.run()
        experiment.p.output_result(f'{out_path}xfc_{alpha}_{threshold}_{MAXITER}_{xfc_ratio}.csv')

def compare_xfc_exe_time(root_folder = os.getcwd(), xfc_ratios = [0.1, 0.2, 0.3, 0.4, 0.5], proning = 10, out_alias = 'xfc_exe_time_vary_ratio.json'):
    print(f'running all xfc with alpha = {ALPHA}, threshold = {THRESHOLD}, maxIter = {MAXITER}, xfc_ratio = {xfc_ratios}')
    iteration_times = {}
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        for xfc_ratio in xfc_ratios:
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = proning)
            experiment.p.determine_xfc(xfc_ratio)
            exe_time = experiment.run()
            iteration_times[xfc_ratio] = exe_time
        if 0 in xfc_ratios:
            pass
        else:
            # add no_xfc baseline
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=False, verbose=False)
            exe_time = experiment.run()
            iteration_times[0] = exe_time

        # store the execution time to the log file
        with open(out_path + out_alias, 'w') as f:
            json.dump(iteration_times, f)

# def compare_xfc_exe_time_vary_proning(root_folder = os.getcwd(), xfc_ratio = 0.1, pronings = [10, 20, 30, 40, 50], out_alias = 'xfc_exe_time_vary_proning.json'):
#     iteration_times = {}
#     for path in os.listdir(root_folder + '/inputs'):
#         in_path = root_folder + '/inputs/' + path + '/'
#         out_path = root_folder + '/outputs/' + path + '/'
#         for proning in pronings:
#             experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = proning)
#             experiment.p.determine_xfc(xfc_ratio)
#             exe_time = experiment.run()['time_per_iteration']
#             iteration_times[proning] = exe_time
#         if 0 in pronings:
#             pass
#         else:
#             # add no proning baseline
#             experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = 0)
#             experiment.p.determine_xfc(xfc_ratio)
#             exe_time = experiment.run()['time_per_iteration']
#             iteration_times[0] = exe_time

#         # store the execution time to the log file
#         with open(out_path + out_alias, 'w') as f:
#             json.dump(iteration_times, f)

#     # def forgot_base_case(root_folder = os.getcwd()):
#     # for path in os.listdir(root_folder + '/inputs'):
#     #     in_path = root_folder + '/inputs/' + path + '/'
#     #     out_path = root_folder + '/outputs/' + path + '/'
#     #     curr_dict = json.load(open(out_path + 'xfc_exe_time_3.json', 'r'))
#     #     experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = 0)
#     #     experiment.p.determine_xfc(0.1)
#     #     exe_time = experiment.run()['time_per_iteration']
#     #     curr_dict['0'] = exe_time

#     #     with open(out_path + 'xfc_exe_time_3.json', 'w') as f:
#     #         json.dump(curr_dict, f)


# def compare_xfc_iterations_vary_proning(root_folder = os.getcwd(), xfc_ratio = 0.1, pronings = [10, 20, 30, 40, 50], out_alias = 'xfc_iterations_vary_proning.json'):
#     iterations = {}
#     for path in os.listdir(root_folder + '/inputs'):
#         in_path = root_folder + '/inputs/' + path + '/'
#         out_path = root_folder + '/outputs/' + path + '/'
#         for proning in pronings:
#             experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = proning)
#             experiment.p.determine_xfc(xfc_ratio)
#             iteartion = experiment.run()['iteration']
#             iterations[proning] = iteartion
#         # add no proning baseline
#         if 0 in pronings:
#             pass
#         else:
#             experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = 0)
#             experiment.p.determine_xfc(xfc_ratio)
#             iteration = experiment.run()['iteration']
#             iterations[0] = iteration

#         # store the execution time to the log file
#         with open(out_path + out_alias, 'w') as f:
#             json.dump(iterations, f)

def compare_xfc_result_vary_proning(root_folder = os.getcwd(), xfc_ratio = XFC_RATIO, pronings = [0, 20, 10, 5, 1], out_alias = 'xfc_results_vary_proning_%s_%s.json', centrality = 'degree'):
    results = {}
    locs = {}
    print(f'running all xfc with alpha = {ALPHA}, threshold = {THRESHOLD}, maxIter = {MAXITER}, xfc_ratio = {xfc_ratio}, proning = {pronings}, centrality = {centrality}')
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        for proning in pronings:
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = proning)
            experiment.p.determine_xfc(xfc_ratio)
            results[proning] = experiment.run()
            locs[proning] = [node.id for node in experiment.p.xfc_set] # type: ignore

        with open(out_path + out_alias % (xfc_ratio, centrality), 'w') as f:
            json.dump(results, f)
            
        with open(out_path + 'xfc_locations_vary_pronings_%s_%s.json' % (xfc_ratio, centrality), 'w+') as f:
            json.dump(locs, f)


def generate_problem(root_folder = os.getcwd(), problem_name = 'Anaheim'):
    path_to_input = root_folder + '/inputs/' + problem_name + '/'
    # create an experiment without outpath
    experiment = Experiment(input_path = path_to_input, output_path='' ,algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=False, verbose=False)
    return experiment.p


def compare_xfc_result_vary_centralities(root_folder = os.getcwd(), xfc_ratio = 0.05, proning = 0, out_alias = 'xfc_result_vary_centralities_%s_%s.json', centralities = ['degree', 'betweenness', 'eigenvector', 'closeness', 'weighted_betweenness', 'adjusted_degree', 'adjusted_betweenness', 'demand_in_out', 'demand_in_out_adj', 'greedy', 'full_greedy'], evaluation = 'total_cost'):
    results = {}
    locs = {}
    print(f'running all xfc vary centralities with alpha = {ALPHA}, threshold = {THRESHOLD}, maxIter = {MAXITER}, xfc_ratio = {xfc_ratio}, centralities = {centralities}')
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        
        # load original results from historical execution
        try:
            _original_dict = json.load(open(out_path + 'xfc_centralities_total_cost', 'r'))
        except:
            _original_dict = {}
        
        for centrality in centralities:
            if centrality in _original_dict:
                results[centrality] = _original_dict[centrality]
                continue
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = proning, network_name = path)
            experiment.p.determine_xfc(xfc_ratio, method=centrality)
            results[centrality] = experiment.run()
            locs[centrality] = [node.id for node in experiment.p.xfc_set] # type: ignore
            
        try:
            original_dict = json.load(open(out_path + out_alias % (xfc_ratio, proning), 'r'))
            for key, value in results.items():
                original_dict[key] = value
        except:
            original_dict = results
            
        with open(out_path + out_alias % (xfc_ratio, proning), 'w') as f:
            json.dump(original_dict, f)
            
        with open(out_path + 'xfc_locations_vary_centralities_%s_%s.json' % (xfc_ratio, proning), 'w') as f:
            json.dump(locs, f)
    
    
def get_xfcs_from_centralities(root_folder = os.getcwd(), xfc_ratio = 0.05, proning = 0, out_alias = 'xfc_centralities_location.json', centralities = ['degree', 'betweenness', 'eigenvector', 'closeness', 'weighted_betweenness', 'adjusted_degree', 'demand_in_out', 'demand_in_out_adj']):
    import networkx as nx
    from matplotlib import pyplot as plt
    
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        print(path)
        for centrality in centralities:
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = proning)
            experiment.p.determine_xfc(xfc_ratio, method=centrality)
            nx_graph = experiment.p.graph.get_networkx_graph()
            colors = ['blue' if experiment.p.graph.node_dict[node].is_XFC() else 'red' for node in nx_graph.nodes]
            sizes = [50 if experiment.p.graph.node_dict[node].is_XFC() else 10 for node in nx_graph.nodes]
            nx.draw_networkx(nx_graph, node_color=colors, node_size=sizes, arrows=True, width=0.1, pos=nx.spring_layout(nx_graph, weight='length', seed=20), with_labels=False)
            plt.savefig(out_path + centrality + '.png')
            plt.cla()
            
            

def get_degrees_stats(root_folder = os.getcwd()):
    import networkx as nx
    from matplotlib import pyplot as plt
    
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        print(path)
        experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False)
        nx_graph = experiment.p.graph.get_networkx_graph()
        degrees = nx_graph.degree
        max_deg = max(degrees, key=lambda x: x[1]) #type: ignore
        deg_lst = [0] * (max_deg[1]+1)
        for node, deg in degrees: # type: ignore
            deg_lst[deg] += 1

        plt.bar(range(len(deg_lst)), deg_lst)
        plt.savefig(out_path + 'degrees_stat.png')
        plt.cla()
        
        
