# Text-Similarity-and-Classification
Given over 250,000 pairs of fake news titles that are classified as 'agreed', 'disagreed', and 'unrelated' for training, we aim at producing high test accuracy models with natural language processing technique. The best models are Support Vector Machine with Classification (SVC) and Passive Aggressive Classification, with up to 86% test accuracy.

## models
To run the best models for classifying test.csv in `Data`. Running through the scripts in:
- `svc.py`
- `passive_aggresive.py`

The TF-IDF vectorization model and the word embedding models is included in the subfolders.
Other supervised models include Naive Bayes classifier, Random Forest classifier, Logistic Regression.
Unsupervised models include feedforward neural network, LSTM, and bidirectional LSTM (to be further optimzed).

## test
To run the models with the train set itself by splitting it into train set and validation. Running through the scripts in:
- `svc_test.py`
- `PAC_test.py`

## submission
Stores the predictions on test.csv which gives the best 3 test accuracy
##### The test accuracies hold the 3rd and 4th place on Kaggle leaderboard.

## data
Holds the data provided, and the corpus generated from the train set

## image
Stores the confusion matrices of the top 2 models, and the plots of cosine similarity of title pairs in a word-embedded form
#
*Please make sure all the required packages listed in `requirement.txt` are installed before running this repository.*
