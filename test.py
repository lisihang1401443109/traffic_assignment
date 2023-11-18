import runner
import loader

if __name__ == '__main__':
    loader.clear_outputs('./outputs/')
    exp = runner.Experiment('./inputs/ChicagoSketch/', './outputs/ChicagoSketch/', xfc=True, algorithm = 'dijkstra', method = 'automatic', maxIter=20   ,  alpha = 0.1, threshold = 0.01)
    exp.p.determine_xfc(0.05)
    exp.run()
    
    # baseline_condition
    runner.run_all_networks()
    
    runner.run_vary_alpha()
    
    runner.run_vary_precisions()
    
    runner.run_all_xfc()
    
    runner.compare_xfc_exe_time()
    
    runner.compare_xfc_result_vary_proning()
    
    runner.compare_xfc_result_vary_centralities(
        xfc_ratio = 0.05, proning = 0,
        out_alias = 'xfc_result_vary_centralities_%s_%s.json',
        centralities = ['degree', 'betweenness', 'eigenvector', 'closeness', 'weighted_betweenness', 'adjusted_degree', 'adjusted_betweenness', 'demand_in_out', 'demand_in_out_adj', 'greedy'],
        evaluation = 'total_cost'
    )
    
    # runner.compare_xfc_result_vary_centralities('./', xfc_ratio = 0.05, proning = 0, out_alias = 'xfc_centralities_total_time.json', centralities = ['gurobi'], evaluation = 'total_time')