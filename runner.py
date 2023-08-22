from loader import *
from model import Problem

ALPHA = 0.15
THRESHOLD = 0.001
MAXITER = 100

class Experiment:
    p : Problem

    def __init__(self, input_path: str, output_path: str, algorithm = 'dijkstra', method = 'automatic', xfc = [], alpha=0.15, threshold=0.05, maxIter = 20) -> None:
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

    def __str__(self) -> str:
        return f'Experiment({self.input_path=}\n{self.output_path=}\n{self.algorithm=}\n{self.method=}\n{self.alpha=}\n{self.threashold=}\n{self.maxIter=}\n{self.xfc=}\n)'

    
    def run(self):
        print(self)
        # print(self.p)
        log = self.p.run(algorithm=self.algorithm, alpha=self.alpha, threshold= self.threashold, maxIter = self.maxIter, method = self.method, xfc=self.xfc)
        self.p.output_result(f'{self.output_path}{self.algorithm}_{self.method}_{self.alpha}.csv', log)

def run_all_networks(root_folder = os.getcwd()) -> None:
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', method = 'automatic', alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER)
        experiment.run()

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