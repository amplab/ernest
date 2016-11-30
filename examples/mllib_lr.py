#!/bin/python

import sys
import time

from pyspark import SparkContext
from pyspark.sql import SQLContext

from pyspark.ml.classification import LogisticRegression

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sample_frac = float(sys.argv[1])
    else:
        sample_frac = 1.0

    if len(sys.argv) > 2:
        num_parts = int(sys.argv[2])
    else:
        num_parts = 256

    sc = SparkContext(appName="LogisticRegressionWithElasticNet")
    sc.setLogLevel("WARN")
    sqlContext = SQLContext(sc)

    # Load training data
    training = sqlContext.read.format("libsvm").load("s3n://ernest-data/rcv1_test_256.binary")
    training = training.sample(False, sample_frac).coalesce(num_parts)
    training.cache().count()

    lr = LogisticRegression(maxIter=10, elasticNetParam=0.8)

    start = time.time()
    # Fit the model
    lrModel = lr.fit(training)
    end = time.time()

    print "LR sample: ", sample_frac, " took ", (end-start)
