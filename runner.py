from loader import *
from model import Problem
import time
import json

ALPHA = 0.15
THRESHOLD = 0.001
MAXITER = 100

class Experiment:
    p : Problem

    def __init__(self, input_path: str, output_path: str, algorithm = 'dijkstra', method = 'automatic', xfc = [], alpha=0.15, threshold=0.05, maxIter = 20, verbose = True, proning = 0) -> None:
        G, D = create_graph_and_demands_from_inputs(*list(load_from_folder(input_path)))
        self.input_path = input_path
        self.output_path = output_path
        self.p = Problem(G, D, xfc)
        self.algorithm = algorithm
        self.method = method
        self.alpha = alpha
        self.threashold = threshold
        self.maxIter = maxIter
        self.xfc = xfc
        self.verbose = verbose
        self.proning = proning

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


def run_all_networks(root_folder = os.getcwd()) -> None:
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER)
        experiment.run()
        experiment.p.output_result(f'{out_path}{experiment.algorithm}_{experiment.method}_{experiment.alpha}.csv')

def run_vary_precisions(root_folder = os.getcwd(), precisions = [0.1, 0.01, 0.001, 0.0001]) -> None:
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        for precision in precisions:
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=precision, maxIter = MAXITER)
            experiment.run()


def run_vary_alpha(root_folder = os.getcwd(), alphas = [0.5, 0.4, 0.3, 0.2, 0.1]) -> None:
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        for alpha in alphas:
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'constant', alpha=alpha, threshold=THRESHOLD, maxIter = MAXITER)
            experiment.run()


def run_all_xfc(root_folder = os.getcwd(), alpha=0.01, xfc_ratio = 0.05) -> None:
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=alpha, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False)
        experiment.p.determine_xfc(xfc_ratio)
        experiment.run()
        experiment.p.output_result(f'{out_path}xfc_{alpha}_{xfc_ratio}.csv')

def compare_xfc_exe_time(root_folder = os.getcwd(), xfc_ratios = [0.1, 0.2, 0.3, 0.4, 0.5], proning = 10, out_alias = 'xfc_exe_time.json'):
    iteration_times = {}
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        for xfc_ratio in xfc_ratios:
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = proning)
            experiment.p.determine_xfc(xfc_ratio)
            exe_time = experiment.run()['time_per_iteration']
            iteration_times[xfc_ratio] = exe_time
        if 0 in xfc_ratios:
            pass
        else:
            # add no_xfc baseline
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=False, verbose=False)
            exe_time = experiment.run()['time_per_iteration']
            iteration_times[0] = exe_time

        # store the execution time to the log file
        with open(out_path + out_alias, 'w') as f:
            json.dump(iteration_times, f)

def compare_xfc_exe_time_vary_proning(root_folder = os.getcwd(), xfc_ratio = 0.1, pronings = [10, 20, 30, 40, 50], out_alias = 'xfc_exe_time.json'):
    iteration_times = {}
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        for proning in pronings:
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = proning)
            experiment.p.determine_xfc(xfc_ratio)
            exe_time = experiment.run()['time_per_iteration']
            iteration_times[proning] = exe_time
        if 0 in pronings:
            pass
        else:
            # add no proning baseline
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = 0)
            experiment.p.determine_xfc(xfc_ratio)
            exe_time = experiment.run()['time_per_iteration']
            iteration_times[0] = exe_time

        # store the execution time to the log file
        with open(out_path + out_alias, 'w') as f:
            json.dump(iteration_times, f)

    # def forgot_base_case(root_folder = os.getcwd()):
    # for path in os.listdir(root_folder + '/inputs'):
    #     in_path = root_folder + '/inputs/' + path + '/'
    #     out_path = root_folder + '/outputs/' + path + '/'
    #     curr_dict = json.load(open(out_path + 'xfc_exe_time_3.json', 'r'))
    #     experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = 0)
    #     experiment.p.determine_xfc(0.1)
    #     exe_time = experiment.run()['time_per_iteration']
    #     curr_dict['0'] = exe_time

    #     with open(out_path + 'xfc_exe_time_3.json', 'w') as f:
    #         json.dump(curr_dict, f)


def compare_xfc_iterations_vary_proning(root_folder = os.getcwd(), xfc_ratio = 0.1, pronings = [10, 20, 30, 40, 50], out_alias = 'xfc_iterations.json'):
    iterations = {}
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        for proning in pronings:
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = proning)
            experiment.p.determine_xfc(xfc_ratio)
            iteartion = experiment.run()['iteration']
            iterations[proning] = iteartion
        # add no proning baseline
        if 0 in pronings:
            pass
        else:
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = 0)
            experiment.p.determine_xfc(xfc_ratio)
            iteration = experiment.run()['iteration']
            iterations[0] = iteration

        # store the execution time to the log file
        with open(out_path + out_alias, 'w') as f:
            json.dump(iterations, f)

def compare_xfc_result_vary_proning(root_folder = os.getcwd(), xfc_ratio = 0.1, pronings = [0, 10, 5, 1], out_alias = 'xfc_results.json'):
    results = {}
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        for proning in pronings:
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False, proning = proning)
            experiment.p.determine_xfc(xfc_ratio)
            results[proning] = experiment.run()['total_cost']

        with open(out_path + out_alias, 'w') as f:
            json.dump(results, f)


def generate_problem(root_folder = os.getcwd(), problem_name = 'Anaheim'):
    path_to_input = root_folder + '/inputs/' + problem_name + '/'
    # create an experiment without outpath
    experiment = Experiment(input_path = path_to_input, output_path='' ,algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=False, verbose=False)
    return experiment.p
