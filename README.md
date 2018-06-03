Description
-----------
![](https://img.shields.io/badge/Python-2.7-brightgreen.svg)
![](https://img.shields.io/badge/Scikit-learn-0.19.1-yellowgreen.svg)

This project is a collection of researches related to **sentiment attitudes
extraction**.  Given a mass media russian news articles and list of marked
named entities in it, we may predict a sentiment attitudes -- pos/neg relations
between entities.  Each attitude could be classified as follows: positive,
negative, or neutral.  Having a train and test collections of
[inosmi.ru](https://inosmi.ru) news as a part of 'data.zip' collection, we
might applly and compare a different (feature based) machine learning
approaches, such as svm, nb, rf, knn.

Being applyed for unlabeled articles (test collections), we interested only in
non neutral attitudes. As a result, we extract positive and negative attitudes
and discard neutrals.

Dataset
-------

We use [RuSentRel 1.0](https://github.com/nicolay-r/RuSentRel/tree/v1.0/)
corpus consisted of analytical articles from Internet-portal
inosmi.ru.


Results
-------

Presented in table below (in comparison with baseline neg/pos/distr methods)

| Model               | Precision | Recall   | F1(P,N)  |
|--------------------:|:---------:|:--------:|:--------:|
|Baseline neg         |  0.03     | 0.39     | 0.05     |
|Baseline pos         |  0.02     | 0.40     | 0.04     |
|Baseline distr       |  0.05     | 0.23     | 0.08     |
|KNN                  |  0.18     | 0.06     | 0.09     |
|SVM (GRID)           |  0.09     | **0.36** | 0.15     |
|Random forest (GRID) |  **0.41** | 0.21     | **0.27** |

Installation
------------

Using [virtualenv](https://www.pythoncentral.io/how-to-install-virtualenv-python/).
Create virtual environment, suppose `my_env`, and activate it as follows:
```
virtualenv my_env
source my_env/bin/activate
```

Use `Makefile` to install [core](https://github.com/nicolay-r/sentiment-erc-core)
library and download [RuSentRel 1.0](https://github.com/nicolay-r/RuSentRel/tree/v1.0/)
dataset:
```
make install
```

We use word2vec model which were taken from
[rusvectores](http://rusvectores.org/static/models/rusvectores2/),
Because of some features that depends on word embedding vocabulary, it is
necessary to additionally download a model as follows:
```
make download_model
```
*Note:* This word embedding model stores a russian terms with additional POS
suffix written in [mystem notation](https://tech.yandex.ru/mystem/doc/grammemes-values-docpage/).

Usage
-----
Being not interested in neutral attitutes, dataset doesn't provide such
information (i.e. attitudes with 'neutral' labels).
For extraction of positive and negative attitudes we additionally introduce
(extract from news) **neutral attudes** to distinguish really sentiment
attitudes from neutral in further.

At first, we compose a list of neutral relations per each news of train an test
collection by running:
```
./neutrals.py
```

Next, compose a list of feature vectors per each attitude of test and train
collection as follows:
```
./vectorize.py
```

Finally we are ready to apply different models by calling:
```
./predict_*.py
```
Where asterics sign ```*``` denotes a pattern matching and group of methods you want,
and here could be `default` (which also includes
[grid](http://scikit-learn.org/stable/modules/grid_search.html) search),
`class_elemination`, `mfs` (model features selection), `rfe` (recursive
features selection), `uv` (univariate)
