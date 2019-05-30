import argparse
from hbconfig import Config
import tensorflow as tf
import data_loader
import numpy as np
from model import Model
import sys
import data_cleaner

from flask import Flask, request
from flask_restful import Resource, Api 
#import keras
import os
import nltk
import json
#from keras.preprocessing.text import Tokenizer
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.models import model_from_yaml
#from keras import backend as K
# load_dictionary_and_weights.py is seperate file to load word2index dictionary and list of classes from files

# Create Flask-Restful API
app = Flask(__name__)
api = Api(app)
# Class for base url "/"

def predict(ids):
    X = np.array(data_loader._pad_input(ids, Config.data.max_seq_length), dtype=np.int32)
    X = np.reshape(X, (1, Config.data.max_seq_length))

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"input_data": X},
            num_epochs=1,
            shuffle=False)

    estimator = _make_estimator()
    result = estimator.predict(input_fn=predict_input_fn)
    prediction = next(result)["prediction"]
    return prediction


def _make_estimator():
    params = tf.contrib.training.HParams(**Config.model.to_dict())
    # Using CPU
    run_config = tf.contrib.learn.RunConfig(
        model_dir=Config.train.model_dir,
        session_config=tf.ConfigProto(
            device_count={'GPU': 0}
        ))

    model = Model()
    return tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=Config.train.model_dir,
            params=params,
            config=run_config)



class HelloWorld(Resource):
    def get(self):
        return {'about' : 'Hello World!'}

    def post(self):
        some_json = request.get_json()
        return {'you sent': some_json}, 201

# Class for "/topics" url. This API is for text classification
class GetTopics(Resource):
    def post(self):
        # Extract data and decode it
        contents = request.data
        
        sentence = contents.decode("utf-8") 
        if sentence == '':
            return "Didn't got any data"

        print("\n\n\n Your data=\n",sentence)
        sentence = data_cleaner.clean_data(sentence)
        print("\n\n\n Cleaned data = \n",sentence)

        data_loader.set_max_seq_length(['train_X_ids', 'test_X_ids'])
        vocab = data_loader.load_vocab("vocab")
        Config.data.vocab_size = len(vocab)
        ids = data_loader.sentence2id(vocab, sentence)
        #print("Sentence = ", sentence)
        #print("IDs = ",ids)
        if len(ids) > Config.data.max_seq_length:
            print(f"Max length I can handle is: {Config.data.max_seq_length}")
            

        result = predict(ids)
        print(result)
        f = open("class_list.txt","r")
        jsonfied_classes_list = f.read()
        classes = json.loads(jsonfied_classes_list)
        print(classes[result])

        return classes[result]
    
    

# Add resource classes to API   
api.add_resource(HelloWorld, '/')
api.add_resource(GetTopics,'/topics')
# 
if __name__ == '__main__':
    #app.run(debug=True)
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='kaggle_movie_review',
                        help='config file name')
    args = parser.parse_args()

    Config(args.config)
    Config.model.batch_size = 1

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)

    app.run(host= '0.0.0.0',port=5001, debug= True)
