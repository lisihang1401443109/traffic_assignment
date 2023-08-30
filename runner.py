from loader import *
from model import Problem
import time
import json

ALPHA = 0.15
THRESHOLD = 0.001
MAXITER = 100

class Experiment:
    p : Problem

    def __init__(self, input_path: str, output_path: str, algorithm = 'dijkstra', method = 'automatic', xfc = [], alpha=0.15, threshold=0.05, maxIter = 20, verbose = True) -> None:
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

    def __str__(self) -> str:
        return f'Experiment({self.input_path=}\n{self.output_path=}\n{self.algorithm=}\n{self.method=}\n{self.alpha=}\n{self.threashold=}\n{self.maxIter=}\n{self.xfc=}\n)'

    
    def run(self):
        print(self)
        # print(self.p)
        log = self.p.run(algorithm=self.algorithm, alpha=self.alpha, threshold= self.threashold, maxIter = self.maxIter, method = self.method, xfc=self.xfc, verbose = self.verbose)
        # self.p.output_result(f'{self.output_path}{self.algorithm}_{self.method}_{self.alpha}.csv', log)

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

def compare_xfc_exe_time(root_folder = os.getcwd(), xfc_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]):
    exe_times = {}
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        for xfc_ratio in xfc_ratios:
            experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=True, verbose=False)
            experiment.p.determine_xfc(xfc_ratio)
            exe_time = experiment.exe_time()
            exe_times[xfc_ratio] = exe_time
        # add no_xfc baseline
        experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER, xfc=False, verbose=False)
        exe_time = experiment.exe_time()
        exe_times[0] = exe_time

        # store the execution time to the log file
        with open(out_path + 'xfc_exe_time.json', 'w') as f:
            json.dump(exe_times, f)

