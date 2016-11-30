import numpy as np
import scipy
from scipy.optimize import nnls
import csv
import sys

class Predictor(object):

  def __init__(self, training_data_in=[], data_file=None):
    ''' 
        Initiliaze the Predictor with some training data
        The training data should be a list of [mcs, input_fraction, time]
    '''
    self.training_data = []
    self.training_data.extend(training_data_in)
    if data_file:
      with open(data_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
          if row[0][0] != '#':
            parts = row[0].split(',')
            mc = int(parts[0])
            scale = float(parts[1])
            time = float(parts[2])
            self.training_data.append([mc, scale, time])

  def add(self, mcs, input_fraction, time):
    self.training_data.append([mcs, input_fraction, time])

  def predict(self, input_fraction, mcs):
    ''' 
        Predict running time for given input fraction, number of machines.
    '''    
    test_features = np.array(self._get_features([input_fraction, mcs]))
    return test_features.dot(self.model[0])

  def predict_all(self, test_data):
    ''' 
        Predict running time for a batch of input sizes, machines.
        Input test_data should be a list where every element is (input_fraction, machines)
    '''    
    test_features = np.array([self._get_features([row[0], row[1]]) for row in test_data])
    return test_features.dot(self.model[0])

  def fit(self):
    print "Fitting a model with ", len(self.training_data), " points"
    labels = np.array([row[2] for row in self.training_data])
    data_points = np.array([self._get_features(row) for row in self.training_data])
    self.model = nnls(data_points, labels)
    # TODO: Add a debug logging mode ?
    # print "Residual norm ", self.model[1]
    # print "Model ", self.model[0]
    # Calculate training error
    training_errors = []
    for p in self.training_data:
      predicted = self.predict(p[0], p[1])
      training_errors.append(predicted / p[2])

    print "Average training error %f%%" % ((np.mean(training_errors) - 1.0)*100.0 )
    return self.model[0]

  def num_examples(self):
    return len(self.training_data)

  def _get_features(self, training_point):
    mc = training_point[0]
    scale = training_point[1]
    return [1.0, float(scale) / float(mc), float(mc), np.log(mc)]

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print "Usage <predictor.py> <csv_file_train>"
    sys.exit(0)

  pred = Predictor(data_file=sys.argv[1])

  model = pred.fit()
  
  test_data = [[i, 1.0] for i in [16, 32, 48, 64, 96, 128]]
  # test_data = []
  # for i in xrange(1, 29):
  #   test_data.append([1.0, i])
  predicted_times = pred.predict_all(test_data)
  print
  print "[Machines, Scale] Predicted Time"
  for i in xrange(0, len(test_data)):
    print test_data[i], predicted_times[i]
