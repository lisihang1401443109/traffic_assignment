from model import *
from numpy import ndarray
import seaborn as sns
import runner
from runner import ALPHA, THRESHOLD, MAXITER

def matrix_visualization(result : Graph):
    n_nodes = len(result.nodes)
    mat = zeros((n_nodes, n_nodes))
    reversed_index = {node: i for i, node in enumerate(result.nodes)}
    for link in result.linkset:
        mat[reversed_index[link.start], reversed_index[link.end]] = link.flow
        sns.heatmap(mat)

if __name__ == '__main__':
    exp = runner.Experiment(input_path = './inputs/Braess/', output_path = './outputs/', algorithm = 'dijkstra', method = 'automatic', xfc=[3], alpha=ALPHA, threshold=THRESHOLD, maxIter = MAXITER)
    exp.run()

    
