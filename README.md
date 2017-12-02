# text-cnn

This code implements [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) models.

- Figure 1: Illustration of a CNN architecture for sentence classification

![figure-1](images/figure-1.png)


## Requirements

- Python 3.6
- TensorFlow 1.4
- hb-config

## Features

- Using Higher-APIs in TensorFlow
	- [Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
	- [Experiment](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment)
	- [Dataset](https://www.tensorflow.org/api_docs/python/tf/contrib/data/Dataset)


## Todo


## Config

example: cornell-movie-dialogs.yml

```yml
data:
  base_path: 'data/'
  processed_path: 'tiny_processed_data'
  max_seq_length: 30
  num_classes: 2
  PAD_ID: 0

model:
  embed_dim: 32
  num_filters: 16
  filter_sizes:
    - 2
    - 3
    - 4
  dropout: 0.5

train:
  batch_size: 1
  learning_rate: 0.001
  train_steps: 10000
  model_dir: 'logs/check_tiny'
  save_every: 1000
  loss_hook_n_iter: 1
  check_hook_n_iter: 10
  min_eval_frequency: 10
```


## Usage

Install requirements.

```pip install -r requirements.txt```

First, check if the model is valid.

```python main.py --config check_tiny --mode train```

Then, download [Dataset]() and train it.

```
sh prepare_dataset
python main.py --config sentiment_dataset --mode train_and_evaluate
```

### Tensorboard

```tensorboard --logdir logs```


## Reference

- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) by Denny Britz
- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) (2014) by Y Kim
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf) (2015) Y Zhang
