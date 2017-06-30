## Example of running Ernest using Apache Spark ML

This document presents an example of using Ernest to build a performance
model for binary classification using Logistic Regression implemented in [Spark
ML](http://spark.apache.org/mllib).

### Step1: Dataset, Experiment Design

For this example we will use the [RCV1
dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#rcv1.binary) from the
LibSVM repository. A pre-processed version of the dataset (after converting negative labels to 0 as
required by MLlib) is available at `s3://ernest-data/rcv1_test_256.binary`.

The first step in using Ernest is to use the Experiment Design module to figure out what training
data points need to be collected. To do this we can run the following command
```
python expt_design.py --min-parts 4 --max-parts 32 --total-parts 256 --min-mcs 1 --max-mcs 8 --cores-per-mc 4
```

In the above case we choose the minimum and maximum number of data partitions that will be used for 
collecting training data and also set the maximum number of machines we wish to use. Finally since
this tutorial uses `r3.xlarge` instances, we set the `cores-per-mc` as 4. 

The output from running this command looks something like
```
Machines, Cores, InputFraction, Partitions, Weight
8, 32, 0.125000, 32, 1.000000
1, 4, 0.015625, 4, 1.000000
1, 4, 0.021382, 6, 1.000000
...
```

This table shows the training data points we will next collect

### Step 2: Data collection

To collect training data we launch a 8 node cluster of r3.xlarge machines. We can use existing tools
like [spark-ec2](https://github.com/amplab/spark-ec2) to do this.

```
./spark-ec2 -s 8 -t r3.xlarge -i <key-file> -k <key-name> --copy-aws-credentials --spark-version 1.6.2 launch ernest-demo
```

Once the cluster is up, we next run our target application with the sampling fraction and machine
sizes listed above. An example for Logistic Regression with RCV1 is in the file
[mllib_lr.py](mllib_lr.py) and a corresponding script to run this for various configurations is in
[collect_data.sh](collect_data.sh). One important thing to note here is that we only run 10
iterations of the algorithm as that is sufficient for building a model. While training on the
complete data, the number of iterations and other parameters can be tweaked.

After we collect the necessary data we put it together in a CSV file to feed into the model builder.
For the above example the [CSV file](rcv1-parsed.csv) looks as follows
```
#Cores,Input Fraction, Time (s)
32,0.125,7.94516801834
4,0.015625,4.72029209137
4,0.021382,4.87661099434
...
```

### Step 3: Model Building

Our last step is to build the performance model using the collected data and then use it to predict
behavior on large clusters, data sizes. To do this we can run the predictor with a command that
looks like
```
python predictor.py rcv1-parsed.csv
```
This prints the predicted time taken to process the entire dataset when using up to 64 machines and
the output for this case looks like
```
Machines, Predicted Time
4 44.6515640166
8 25.4777295249
12 19.36348049
16 16.4412832993
20 14.7682298198
24 13.7061636865
28 12.9855393036
...
```

Thus what we see is that the model predicts that as we go from 16 to 24 machines, the performance wins
are limited as the time for 10 iterations only drops from 16.4s to 12.98s. This is because RCV1 is a
very small dataset and at larger cluster sizes we spend more time on communication rather than on
parallel computation. [Our paper](http://shivaram.org/publications/ernest-nsdi.pdf) contains more 
examples.
