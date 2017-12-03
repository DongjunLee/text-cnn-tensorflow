# text-cnn

This code implements [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) models.

- Figure 1: Illustration of a CNN architecture for sentence classification

![figure-1](images/figure-1.png)


## Requirements

- Python 3.6
- TensorFlow 1.4
- hb-config
- tqdm

## Features

- Using Higher-APIs in TensorFlow
	- [Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
	- [Experiment](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment)
	- [Dataset](https://www.tensorflow.org/api_docs/python/tf/contrib/data/Dataset)
- Dataset : [rt-polarity](https://github.com/yoonkim/CNN_sentence), [Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)

## Todo

- apply embed_type 
	- CNN-rand
	- CNN-static
	- CNN-nonstatic
	- CNN-multichannel


## Config

example: kaggle\_movie\_review.yml

```yml
data:
  type: 'kaggle_movie_review'
  base_path: 'data/'
  raw_data_path: 'kaggle_movie_reviews/'
  processed_path: 'kaggle_processed_data'
  testset_size: 25000
  num_classes: 5
  PAD_ID: 0

model:
  embed_type: 'rand'  (rand, static, non-static, multichannel)
  pretrained_embed: "" 
  embed_dim: 300
  num_filters: 256
  filter_sizes:
    - 2
    - 3
    - 4
    - 5
  dropout: 0.5

train:
  batch_size: 64
  learning_rate: 0.00005
  train_steps: 100000
  model_dir: 'logs/kaggle_movie_review'
  save_checkpoints_steps: 2000
  check_hook_n_iter: 1000
  min_eval_frequency: 1000
```


## Usage

Install requirements.

```pip install -r requirements.txt```

Then, prepare dataset and train it.

```
sh prepare_kaggle_movie_reviews.sh
python main.py --config kaggle_movie_review --mode train_and_evaluate
```

### Tensorboard

```tensorboard --logdir logs```

- Category Color

![category_image](images/category.png)

- rt-polarity (binary classification)

![images](images/rt-polarity_loss_and_accuracy.jpeg)

- kaggle_movie_review (multiclass classification)

![images](images/kaggle-loss_and_accuracy.jpg)


## Reference

- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) by Denny Britz
- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) (2014) by Y Kim
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf) (2015) Y Zhang
