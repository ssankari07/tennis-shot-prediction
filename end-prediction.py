from comet_ml import Experiment
import os,sys
experiment = Experiment(api_key=os.environ["COMET_TENNIS_PREDICTION_KEY"], project_name="tennis-shot-end-prediction")

import pandas as pd
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(pardir,"data/preprocessed/match_data_with_potision.csv"))

##another preprocess for end rows
new_end = []
for idx,row in df.iterrows():
    e = row["end"]
    e = e.upper().replace(" ","")
    e = e.replace("\xa0","")
    new_end.append(e)
df["end"] = new_end

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(df["X"]))
seq = tokenizer.texts_to_sequences(list(df["X"]))
word_index = tokenizer.word_index
num_words = len(word_index)

maxlen = max([len(x) for x in seq])
X = sequence.pad_sequences(seq, maxlen=maxlen)

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
le = LabelEncoder()
y = le.fit_transform(list(df["end"]))
y = np_utils.to_categorical(y)

##split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)

print("X_train dim", X_train.shape)
print("X_test dim", X_test.shape)
print("y_train dim", y_train.shape)
print("y_test dim", y_test.shape)
#
# Embedding＆LSTM
from keras.models import Model
from keras.layers import Input, Dense,Embedding,LSTM

main_input = Input(shape=(X.shape[1],), name='main_input')
embedding1 = Embedding(output_dim=200, input_dim=num_words+1,mask_zero=True, name='embedding1')(main_input)
lstm1 = LSTM(32, name='lstm1')(embedding1)
main_output = Dense(y.shape[1], name='main_output')(lstm1)
model = Model(inputs=main_input, outputs=main_output)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

print("model_summary")
print(model.summary())

model.fit(X_train,y_train,batch_size=20,epochs=20,validation_data=(X_test, y_test))
