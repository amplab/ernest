import numpy as np
import cvxpy as cvx
import argparse

class ExperimentDesign(object):

    MIN_WEIGHT_FOR_SELECTION = 0.3

    '''
    Represents an experiment design object that can be used to setup
    and run experiment design.
    '''
    def __init__(self, parts_min, parts_max, total_parts,
                 mcs_min=1, mcs_max=16, cores_per_mc=2, budget=10.0,
                 num_parts_interpolate=20):
        '''
        Create an experiment design instance.

        :param self: The object being created
        :type self: ExperimentDesign
        :param parts_min: Minimum number of partitions to use in experiments
        :type parts_min: int
        :param parts_max: Maximum number of partitions to use in experiments
        :type parts_max: int
        :param total_parts: Total number of partitions in the dataset
        :type total_parts: int
        :param mcs_min: Minimum number of machines to use in experiments
        :type mcs_min: int
        :param mcs_max: Maximum number of machines to use in experiments
        :type mcs_max: int
        :param cores_per_mc: Cores or slots available per machine
        :type cores_per_mc: int
        :param budget: Budget for the experiment design problem
        :type budget: float
        :param budget: Number of points to interpolate between parts_min and parts_max 
        :type budget: float
        '''
        self.parts_min = parts_min
        self.parts_max = parts_max
        self.total_parts = total_parts
        self.mcs_min = mcs_min
        self.mcs_max = mcs_max
        self.cores_per_mc = cores_per_mc
        self.budget = budget
        self.num_parts_interpolate = num_parts_interpolate

    def _construct_constraints(self, lambdas, points):
        '''Construct non-negative lambdas and budget constraints'''
        constraints = []
        constraints.append(0 <= lambdas)
        constraints.append(lambdas <= 1)
        constraints.append(self._get_cost(lambdas, points) <= self.budget)
        return constraints

    def _get_cost(self, lambdas, points):
        '''Estimate the cost of an experiment. Right now this is input_frac/machines'''
        cost = 0
        num_points = len(points)
        scale_min = float(self.parts_min) / float(self.total_parts)
        for i in xrange(0, num_points):
            scale = points[i][0]
            mcs = points[i][1]
            cost = cost + (float(scale) / scale_min * 1.0 / float(mcs) * lambdas[i])
        return cost

    def _get_training_points(self):
        '''Enumerate all the training points given the params for experiment design'''
        mcs_range = xrange(self.mcs_min, self.mcs_max + 1)

        scale_min = float(self.parts_min) / float(self.total_parts)
        scale_max = float(self.parts_max) / float(self.total_parts)
        scale_range = np.linspace(scale_min, scale_max, self.num_parts_interpolate)

        for scale in scale_range:
            for mcs in mcs_range:
                if np.round(scale * self.total_parts) >= self.cores_per_mc * mcs:
                    yield [scale, mcs]

    def _frac2parts(self, fraction):
        '''Convert input fraction into number of partitions'''
        return int(np.ceil(fraction * self.total_parts))

    def run(self):
        ''' Run experiment design. Returns a list of configurations and their scores'''
        training_points = list(self._get_training_points())
        num_points = len(training_points)

        all_training_features = np.array([_get_features(point) for point in training_points])
        covariance_matrices = list(_get_covariance_matrices(all_training_features))

        lambdas = cvx.Variable(num_points)

        objective = cvx.Minimize(_construct_objective(covariance_matrices, lambdas))
        constraints = self._construct_constraints(lambdas, training_points)

        problem = cvx.Problem(objective, constraints)

        opt_val = problem.solve()
        # TODO: Add debug logging
        # print "solution status ", problem.status
        # print "opt value is ", opt_val

        filtered_lambda_idxs = []
        for i in range(0, num_points):
            if lambdas[i].value > self.MIN_WEIGHT_FOR_SELECTION:
                filtered_lambda_idxs.append((lambdas[i].value, i))

        sorted_by_lambda = sorted(filtered_lambda_idxs, key=lambda t: t[0], reverse=True)
        return [(self._frac2parts(training_points[idx][0]), training_points[idx][0],
                 training_points[idx][1], l) for (l, idx) in sorted_by_lambda]

def _construct_objective(covariance_matrices, lambdas):
    ''' Constructs the CVX objective function. '''
    num_points = len(covariance_matrices)
    num_dim = int(covariance_matrices[0].shape[0])
    objective = 0
    matrix_part = np.zeros([num_dim, num_dim])
    for j in xrange(0, num_points):
        matrix_part = matrix_part + covariance_matrices[j] * lambdas[j]

    for i in xrange(0, num_dim):
        k_vec = np.zeros(num_dim)
        k_vec[i] = 1.0
        objective = objective + cvx.matrix_frac(k_vec, matrix_part)

    return objective

def _get_covariance_matrices(features_arr):
    ''' Returns a list of covariance matrices given expt design features'''
    col_means = np.mean(features_arr, axis=0)
    means_inv = (1.0 / col_means)
    nrows = features_arr.shape[0]
    for i in xrange(0, nrows):
        feature_row = features_arr[i,]
        ftf = np.outer(feature_row.transpose(), feature_row)
        yield np.diag(means_inv).transpose().dot(ftf.dot(np.diag(means_inv)))

def _get_features(training_point):
    ''' Compute the features for a given point. Point is expected to be [input_frac, machines]'''
    scale = training_point[0]
    mcs = training_point[1]
    return [1.0, float(scale) / float(mcs), float(mcs), np.log(mcs)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Design')

    parser.add_argument('--min-parts', type=int, required=True,
        help='Minimum number of partitions to use in experiments')
    parser.add_argument('--max-parts', type=int, required=True,
        help='Maximum number of partitions to use in experiments')
    parser.add_argument('--total-parts', type=int, required=True,
        help='Total number of partitions in the dataset')

    parser.add_argument('--min-mcs', type=int, required=True,
        help='Minimum number of machines to use in experiments')
    parser.add_argument('--max-mcs', type=int, required=True,
        help='Maximum number of machines to use in experiments')

    parser.add_argument('--cores-per-mc', type=int, default=2,
        help='Number of cores or slots available per machine, (default 2)')
    parser.add_argument('--budget', type=float, default=10.0,
        help='Budget of experiment design problem, (default 10.0)')
    parser.add_argument('--num-parts-interpolate', type=int, default=20,
        help='Number of points to interpolate between min_parts and max_parts, (default 20)')

    args = parser.parse_args()

    ex = ExperimentDesign(args.min_parts, args.max_parts, args.total_parts,
        args.min_mcs, args.max_mcs, args.cores_per_mc, args.budget,
        args.num_parts_interpolate)

    expts = ex.run()
    print "Machines, Cores, InputFraction, Partitions, Weight"
    for expt in expts:
        print "%d, %d, %f, %d, %f" % (expt[2], expt[2] * args.cores_per_mc, expt[1], expt[0], expt[3])
