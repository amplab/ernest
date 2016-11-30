## Ernest: Efficient Perfomance Prediction for Advanced Analytics

Ernest is a performance predition framework for analytics jobs developed using frameworks like Apache Spark and run on cloud computing infrastructure. 

One of the main challenges in deploying large scale analytics applications in
the cloud is choosing the right hardware configuration. Specifically in Amazon
EC2 or Google Compute Engine clusters, choosing the right instance type and the
right number of instances can significantly improve performance or lower cost. 

Ernest is a performance modeling framework that helps address this problem.
Ernest builds performance models based on the behavior of the job on small
samples of data and then predict its performance on larger datasets and cluster
sizes. To minimize the time and resources spent in building a model, Ernest
uses [optimal experiment design](https://en.wikipedia.org/wiki/Optimal_design),
a statistical technique that allows us to collect as few training points as
required. For more details please see our [paper]
(http://shivaram.org/publications/ernest-nsdi.pdf) and [talk slides](http://shivaram.org/talks/ernest-nsdi-2016.pdf) from NSDI 2016.

### Installing Ernest

The easiest way to install Ernest is by cloning this repository.

Running Ernest requires installing [SciPy](http://scipy.org), [NumPy](http://numpy.org) and
[CVXPY](http://www.cvxpy.org). An easy way to do this is using the `requirements.txt` file.

```
pip install -r requirements.txt
```

### Using Ernest

We describe at a high level the three main steps to use Ernest. You can also see our walk through for an example from Spark MLlib

1. Determining what sample data points to collect. To do this we will be using experiment design implemented in [expt_design.py](expt_design.py). This will return the set of training data points required to build a performance model.  
2. Collect running time for the set of training data points. These can be executed using [Spark EC2 scripts](http://github.com/amplab/spark-ec2) or Amazon EMR etc.
3. Building a performance model and using it for prediction. To do this we create a CSV file with measurements from previous step and use [predictor.py](predictor.py). 
