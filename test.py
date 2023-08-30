import runner
import loader

if __name__ == '__main__':
    loader.clear_outputs('./outputs/')
    exp = runner.Experiment('./inputs/ChicagoSketch/', './outputs/ChicagoSketch/', xfc=True, algorithm = 'dijkstra', method = 'automatic', maxIter=20   ,  alpha = 0.1, threshold = 0.01)
    exp.p.determine_xfc(0.05)
    exp.run()