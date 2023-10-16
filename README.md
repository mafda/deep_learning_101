# Deep Learning 101

This repository presents the **basic notions** that involve the concept of Machine Learning and Deep Learning.

Read more in this post [ML & DL — Machine Learning and Deep Learning 101](https://medium.com/@mafda_/ml-dl-machine-learning-and-deep-learning-101-2686d93d70d).

## Configure environment

- Create the conda environment

```shell
(base)$: conda env create -f environment.yml
```

- Activate the environment

```shell
(base)$: conda activate deep_learning_101
```

- Run!

```shell
(deep_learning_101)$: python -m jupyter notebook
```

## Models

The models include a brief theoretical introduction and practical implementations developed using Python and Keras/TensorFlow in Jupyter Notebooks.

### Development Environment:

* [ML & DL — Development environment (Part 1)](https://mafda.medium.com/ml-dl-development-environment-part-1-5bb0b35750aa)

### Theoretical introduction (https://mafda.medium.com):

* [ML & DL — Linear Regression (Part 2)](https://mafda.medium.com/ml-dl-linear-regression-part-2-14f114f2d62a)
* [ML & DL — Logistic Regression (Part 3)](https://mafda.medium.com/ml-dl-logistic-regression-part-3-fe6aca8f01b)
* [ML & DL — Artificial Neural Networks (Part 4)](https://mafda.medium.com/ml-dl-artificial-neural-networks-part-4-619350a93ef1)
* [ML & DL — Deep Neural Networks (Part 5)](https://mafda.medium.com/ml-dl-deep-artificial-neural-networks-part-5-568ad05be712)
* [ML & DL — Convolutional Neural Networks (Part 6)](https://mafda.medium.com/ml-dl-convolutional-neural-networks-part-6-97357db58165)

### Practical implementations (Jupyter Notebooks):

* [Linear Regression](https://github.com/mafda/deep_learning_101/blob/master/src/01-linear-regression.ipynb)
* [Logistic Regression](https://github.com/mafda/deep_learning_101/blob/master/src/02-logistic-regression.ipynb)
* [Artificial Neural Networks](https://github.com/mafda/deep_learning_101/blob/master/src/03-artificial-neural-networks.ipynb)
* [Deep Neural Networks](https://github.com/mafda/deep_learning_101/blob/master/src/04-deep-neural-networks.ipynb)
* [Convolutional Neural Networks](https://github.com/mafda/deep_learning_101/blob/master/src/05-convolutional-neural-networks.ipynb)

### Results

| Model | Architecture | Activation | Parameters| Accuracy |
| ----: | ----: | ----: | ----: | ----: |
| Log Reg | -- | -- | 7850 | 92.79% |
| ANN | [32] | [sigmoid] | 25450| 96.27% |
| DNN | [128, 64] | [relu, relu] | 25450 | 97.90% |
| CNN | [32, 64, 128] | [relu, relu, relu] | 25450 | 98.84% |

## [pt-BR] Presentation

* [deep-learning-101.pdf](https://github.com/mafda/deep_learning_101/blob/master/pdf/deep-learning-101.pdf)

## References

* Complete Post Medium
  * [ML & DL — Machine Learning and Deep Learning 101](https://mafda.medium.com/ml-dl-machine-learning-and-deep-learning-101-2686d93d70d)

* Book
  * [Deep Learning Book](http://www.deeplearningbook.org/)

---

made with 💙 by [mafda](https://mafda.github.io/)
