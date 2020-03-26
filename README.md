# Introduction

## About Course

In this course, you will learn the absolute basics of Data Science and Machine learning. We will go through commonly used terms and write some code in Python

## Prerequisites

None

## What is Data Science and Machine Learning

Machine learning is a field of computer science that **gives computer the ability to learn** without being explicitly programmed

In traditional programing languages, we feed in data (also called *features*) with the predefined algorithm to get the output. However, in machine learning programming we feed in data with the ouput (also called *labels*) and let the model learn and find out the values that fits with the input and output data

![](https://i.ibb.co/NxdRw8n/image.png)

### Types of Machine Learning algorithms

**Supervised machine learning** is the type of machine learning approach where we injest the labeled datasets. For example, detecting if the email is spam or ham. This type of algorithm is mostly used for classification problems.

**Unsupervised machine learning** is the type of machine learning approach where we inject the unlabled datasets and let the model find the patterns in the data. For example, 


**Note:** Your inputs should be relevant to the machine learning algorithm you are choosing

## Other Resources
+ [Python Bootcamp](https://github.com/Pierian-Data/Complete-Python-3-Bootcamp)
+ [The Numpy Stack](https://github.com/tbhaxor/deep-learning-prerequisites)

# Getting Setup with Tools

## Installing Anaconda

+ #### [Windows](https://docs.anaconda.com/anaconda/install/windows/)
+ #### [Linux](https://docs.anaconda.com/anaconda/install/linux/)
+ #### [MacOS](https://docs.anaconda.com/anaconda/install/mac-os/)

## Scikit-Learn

Scikit-Learn is a statistical machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines.

Importing scikit-learn
```python
import sklearn as sk
```

Documentation: https://scikit-learn.org/stable/user_guide.html

## Types of Algorithms

### Supervised
+ Linear Regression
+ Logistic Regression
+ K Nearest Neighbours
+ Decision Trees
+ Naive Bayes

### Unsupervised
+ Centroid-based algorithms
+ Connectivity-based algorithms
+ Density-based algorithms
+ Probabilistic

## Playing with iris data

About Iris: https://archive.ics.uci.edu/ml/datasets/iris

Importing the modules


```python
from sklearn import datasets
```

Loading dataset


```python
iris_data = datasets.load_iris()
type(iris_data)
```




    sklearn.utils.Bunch



This is basically a [`dict`](https://docs.python.org/3/tutorial/datastructures.html#dictionaries), therefore using _.keys()_ to find out the keys


```python
iris_data.keys()
```




    dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])



|Keyname|Description|
|---|---|
|data|features|
|target|normalized label in number|
|target_names|name of labels in string|
|DESCR|description of the dataset (same as [here](https://archive.ics.uci.edu/ml/datasets/iris_))|
|feature_names|names of features|
|filename|filelocation of dataset|


Getting all the feature names


```python
iris_data.feature_names
```




    ['sepal length (cm)',
     'sepal width (cm)',
     'petal length (cm)',
     'petal width (cm)']



Getting all the target names


```python
iris_data.target_names
```




    array(['setosa', 'versicolor', 'virginica'], dtype='<U10')



# Getting Deeper into Machine Learning Frameworks and Algorithms

## K-Nearest Neighbours

The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.

> Birds of a feather flock together.

### Working of KNN

1. Load the data
2. Initialize K to your chosen number of neighbors
3. For each example in the data
     1. Calculate the distance between the query example and the current example from the data.
     2. Add the distance and the index of the example to an ordered collection
4. Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
5. Pick the first K entries from the sorted collection
6. Get the labels of the selected K entries
7. If regression, return the mean of the K labels
8. If classification, return the mode of the K labels

Let's use the KNN in the code


```python
from sklearn.neighbors import KNeighborsClassifier
```

Since our problem is of classification, using `KNeighborsClassifier`

Instancing KNN model with `K=3`


```python
knn = KNeighborsClassifier(n_neighbors=3)
```

So, 
+ **`iris_data.data`** &rarr; features (denoted by `X`) 
+ **`iris_data.target`** &rarr; labels (denoted by `y`)

Before actually fitting you should split the data into what we say **training** and **testing**. And while training, the testing data should not be touched, and vice-versa.

This is done to actually check whether the model is performing well or not


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, shuffle=True, train_size=0.6)
```

The training data will be 60% of the original data and testing data will be 40% of the original data. The `shuffle=True` is used to shuffle the data so that the model doesn't not memorize the pattern. This helps in increasing accuracy


```python
X_train.shape
```




    (90, 4)




```python
X_test.shape
```




    (60, 4)



Now let's fit the model


```python
knn.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                         weights='uniform')



Making predictions


```python
pred = knn.predict([X_test[0]])
pred
```




    array([2])



Getting the label name from the prediction


```python
iris_data.target_names[pred[0]]
```




    'virginica'



Testing if the predicted model is correct of not


```python
iris_data.target_names[y_test[0]]
```




    'virginica'



Now to find the score of the model, simply use `.score()` function


```python
knn.score(X_test, y_test)
```




    0.9166666666666666



The accuracy of our model is 91.66%

## Evaluating and Enhancing Models

Now how would you find if this is the best model or not? Does it seems frustrating.

Luckily the sklearn provides Cross validators to check and find the best model


```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
```

Creating a fresh model


```python
knn = KNeighborsClassifier()
```

Defining hyper parameters (n_neighbors is a type of hyper parameter)


```python
params = {
    "n_neighbors": [*range(1, 11)] # from 1 - 10
}
```

Creating a GridSearchCV object with 3 cross validations per dataset


```python
gcv = GridSearchCV(knn, params, cv=3, verbose=True)
```

Fitting the model


```python
gcv.fit(X_test, y_test)
```

    Fitting 3 folds for each of 10 candidates, totalling 30 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed:    0.1s finished





    GridSearchCV(cv=3, error_score=nan,
                 estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                                metric='minkowski',
                                                metric_params=None, n_jobs=None,
                                                n_neighbors=5, p=2,
                                                weights='uniform'),
                 iid='deprecated', n_jobs=None,
                 param_grid={'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=True)



Getting the best model


```python
gcv.best_estimator_
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=6, p=2,
                         weights='uniform')



The parameters of the model


```python
gcv.best_params_
```




    {'n_neighbors': 6}



As you can see it is different from what we have used earlier (_n_neighbors=3_)

Getting the best score of the model


```python
gcv.best_score_
```




    0.9666666666666667



The score is indeed increased from 91%
