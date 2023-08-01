from loader import *
from model import Problem

class Experiment:
    p : Problem

    def __init__(self, input_path: str, output_path: str, algorithm = 'dijkstra', iteration = 'constant', alpha=0.15, threshold=500, maxIter = 100) -> None:
        G, D = create_graph_and_demands_from_inputs(*list(load_from_folder(input_path)))
        self.input_path = input_path
        self.output_path = output_path
        self.p = Problem(G, D)
        self.algorithm = algorithm
        self.iteration = iteration
        self.alpha = alpha
        self.threashold = threshold
        self.maxIter = maxIter

    def __str__(self) -> str:
        return f'Experiment({self.input_path=}, {self.output_path=}, {self.algorithm=}, {self.iteration=}, {self.alpha=}, {self.threashold=}, {self.maxIter=})'

    
    def run(self):
        print(self)
        self.p.run(algorithm=self.algorithm, alpha=self.alpha, threshold= self.threashold, maxIter = self.maxIter)
        self.p.output_result(self.output_path + self.algorithm + '_' + self.iteration + '.csv')

def run_all_networks(root_folder = os.getcwd()) -> None:
    for path in os.listdir(root_folder + '/inputs'):
        in_path = root_folder + '/inputs/' + path + '/'
        out_path = root_folder + '/outputs/' + path + '/'
        experiment = Experiment(input_path = in_path, output_path = out_path, algorithm = 'dijkstra', iteration = 'constant', alpha=0.15, threshold=500, maxIter = 40)
        experiment.run()