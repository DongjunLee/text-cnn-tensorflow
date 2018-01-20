cd data/kaggle_movie_reviews

unzip train.tsv.zip
unzip test.tsv.zip

cd ../..
python data_loader.py --config kaggle_movie_review
