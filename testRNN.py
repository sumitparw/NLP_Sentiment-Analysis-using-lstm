import nltk
import random
import pandas as pd
from nltk.tokenize import word_tokenize
import string
import re
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,f1_score
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.optimizers import Adam



class rnn():
    word_dict=dict()
    max_cap=80

    def assign_label(self,x):
      if x[2] < 3.0 : return "negative"
      elif x[2] > 3.0 : return "positive"
      else: return "neutral"


    def clean_document(self,doco):
        punctuation = string.punctuation + '\n\n';
        punc_replace = ''.join([' ' for s in punctuation]);
        doco_clean = doco.replace('-', ' ');
        doco_alphas = re.sub(r'\W +', '', doco_clean)
        trans_table = str.maketrans(punctuation, punc_replace);
        doco_clean = ' '.join([word.translate(trans_table) for word in doco_alphas.split(' ')]);
        doco_clean = doco_clean.split(' ');
        doco_clean = [word.lower() for word in doco_clean if len(word) > 0];

        return doco_clean;

    def return_train_test_data_rnn(self,file_path):
                df  = pd.read_csv(file_path,header=None)
                df = df[df.columns[2:4]]
                df[2] = df.apply(self.assign_label, axis=1)
                inx = df[df[2]=='neutral'].index
                df.drop(inx,inplace=True)
                df[2] = df[2].map({'negative': 0, 'positive': 1})
                reviews = np.array(df[3].to_list())
                labels = np.array(df[2].to_list())

                review_cleans = [self.clean_document(doc) for doc in reviews];
                sentences = [' '.join(r) for r in review_cleans]

                tokenizer = Tokenizer();
                tokenizer.fit_on_texts(sentences);
                text_sequences = np.array(tokenizer.texts_to_sequences(sentences));
                sequence_dict = tokenizer.word_index;
                self.word_dict = dict((num, val) for (val, num) in sequence_dict.items());

                reviews_encoded = [];
                for i, review in enumerate(review_cleans):
                    reviews_encoded.append([sequence_dict[x] for x in review]);

                lengths = [len(x) for x in reviews_encoded];
                with plt.xkcd():
                    plt.hist(lengths, bins=range(100))

                max_cap = 80;
                X = pad_sequences(reviews_encoded, maxlen=max_cap, truncating='post')

                Y = np.array([[0,1] if label == 0 else [1,0] for label in labels])
                np.random.seed(1024);
                random_posits = np.arange(len(X))
                np.random.shuffle(random_posits);

                # Shuffle X and Y
                X = X[random_posits];
                Y = Y[random_posits];

                # Divide the reviews into Training, Dev, and Test data.
                train_cap = int(0.70 * len(X));
                dev_cap = int(0.85 * len(X));

                X_train, Y_train = X[:train_cap], Y[:train_cap];
                X_dev, Y_dev = X[train_cap:dev_cap], Y[train_cap:dev_cap];
                X_test, Y_test = X[dev_cap:], Y[dev_cap:]
                return X_train,Y_train,X_dev,Y_dev,X_test,Y_test

    def build_model(self):
                model = Sequential();
                model.add(Embedding(len(self.word_dict), self.max_cap, input_length=self.max_cap));
                model.add(LSTM(80, return_sequences=True, recurrent_dropout=0.2));
                model.add(Dropout(0.2))
                model.add(LSTM(80, recurrent_dropout=0.2));
                model.add(Dense(80, activation='relu'));
                model.add(Dense(2, activation='softmax'));
                print(model.summary());
                return model


    def train_model(self,X_train,Y_train,X_dev,Y_dev):
                model= self.build_model()
                optimizer = Adam(lr=0.01, decay=0.001);
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                # fit model
                model.fit(X_train, Y_train, batch_size=600, epochs=1, validation_data=(X_dev, Y_dev))
                return model

    def predict(self,X_test,model):
                predictions = model.predict_classes(X_test)
                return predictions

    def accuracy(self,predictions,X_test,Y_test,model):
                # Convert Y_test to the same format as predictions
                actuals = [0 if y[0] == 1 else 1 for y in Y_test]

                print("f1_score:"+f1_score)
                # Use SkLearn's Metrics module
                return accuracy_score(predictions, actuals)
