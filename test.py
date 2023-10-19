import runner
import loader

if __name__ == '__main__':
    # loader.clear_outputs('./outputs/')
    # exp = runner.Experiment('./inputs/ChicagoSketch/', './outputs/ChicagoSketch/', xfc=True, algorithm = 'dijkstra', method = 'automatic', maxIter=20   ,  alpha = 0.1, threshold = 0.01)
    # exp.p.determine_xfc(0.05)
    # exp.run()
    
    runner.compare_xfc_result_vary_centralities('./', xfc_ratio = 0.05, proning = 0, out_alias = 'xfc_centralities_total_cost.json', centralities = ['degree', 'betweenness', 'eigenvector', 'closeness', 'weighted_betweenness', 'adjusted_degree', 'adjusted_betweenness', 'greedy', 'full_greedy', 'demand_in_out', 'demand_in_out_adj'], evaluation = 'total_cost')