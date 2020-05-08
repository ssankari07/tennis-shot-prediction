from comet_ml import Experiment
import os, sys
experiment = Experiment(
    api_key=os.environ["COMET_TENNIS_PREDICTION_KEY"],
    project_name="tennis-shot-end-prediction")

import pandas as pd
pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(
    os.path.join(pardir, "data/preprocessed/match_data_with_potision.csv"))

##another preprocess for end rows
new_end = []
for idx, row in df.iterrows():
    e = row["end"]
    e = e.upper().replace(" ", "")
    e = e.replace("\xa0", "")
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

### other data
le_depth = LabelEncoder()
depth = le_depth.fit_transform(list(df["depth"]))
#depth = np_utils.to_categorical(depth)

le_fault = LabelEncoder()
fault = le_fault.fit_transform(list(df["fault"]))
#fault = np_utils.to_categorical(fault)

le_shooter = LabelEncoder()
shooter = le_shooter.fit_transform(list(df["shooter"]))
#shooter = np_utils.to_categorical(shooter)

le_server = LabelEncoder()
server = le_server.fit_transform(list(df["server"]))
#server = np_utils.to_categorical(server)

le_point = LabelEncoder()
point = le_point.fit_transform(list(df["point"]))
#point = np_utils.to_categorical(point)

le_game = LabelEncoder()
game = le_game.fit_transform(list(df["game"]))
#game = np_utils.to_categorical(game)

le_sets = LabelEncoder()
sets = le_sets.fit_transform(list(df["set"]))
#sets = np_utils.to_categorical(sets)

##split
from sklearn.model_selection import train_test_split
X_train,X_test,depth_train,depth_test,fault_train,fault_test,shooter_train,shooter_test, \
server_train,server_test,point_train,point_test,game_train,game_test,sets_train,sets_test,y_train,y_test \
 = train_test_split(X,depth,fault,shooter,server,point,game,sets,y,test_size=0.1,random_state=40)

print("X_train dim", X_train.shape)
print("X_test dim", X_test.shape)
print("y_train dim", y_train.shape)
print("y_test dim", y_test.shape)

# Embedding＆LSTM
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Reshape, Concatenate, Activation

depth_input = Input(shape=(1, ), name='depth_input')
depth_output = Embedding(
    depth.shape[0], 2, name='embedding_depth')(depth_input)
depth_output = Reshape(target_shape=(2, ))(depth_output)

fault_input = Input(shape=(1, ), name='fault_input')
fault_output = Dense(1)(fault_input)

shooter_input = Input(shape=(1, ), name='shooter_input')
shooter_output = Dense(1)(shooter_input)

server_input = Input(shape=(1, ), name='server_input')
server_output = Dense(1)(server_input)

point_input = Input(shape=(1, ), name='point_input')
point_output = Embedding(
    point.shape[0], 10, name='embedding_point')(point_input)
point_output = Reshape(target_shape=(10, ))(point_output)

game_input = Input(shape=(1, ), name='game_input')
game_output = Embedding(game.shape[0], 10, name='embedding_game')(game_input)
game_output = Reshape(target_shape=(10, ))(game_output)

sets_input = Input(shape=(1, ), name='sets_input')
sets_output = Embedding(sets.shape[0], 6, name='embedding_sets')(sets_input)
sets_output = Reshape(target_shape=(6, ))(sets_output)

rally_input = Input(shape=(X.shape[1], ), name='rally_input')
rally_output = Embedding(
    output_dim=50,
    input_dim=num_words + 1,
    mask_zero=True,
    name='embedding_rally')(rally_input)
rally_output = LSTM(32, name='lstm_rally')(rally_output)
rally_output = Dense(10, name='rally_output')(rally_output)

input_model = [
    rally_input,  ##TODO: deal rally data
    depth_input,
    fault_input,
    shooter_input,
    server_input,
    point_input,
    game_input,
    sets_input
]

output_embeddings = [
    rally_output,
    depth_output,
    fault_output,
    shooter_output,
    server_output,
    point_output,
    game_output,
    sets_output,
]

output_model = Concatenate()(output_embeddings)
output_model = Dense(1000, kernel_initializer="uniform")(output_model)
output_model = Activation('relu')(output_model)
output_model = Dense(500, kernel_initializer="uniform")(output_model)
output_model = Activation('relu')(output_model)
output_model = Dense(y.shape[1])(output_model)

model_addCategory = Model(inputs=input_model, outputs=output_model)

model_addCategory.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("model_summary")
print(model_addCategory.summary())

#import pdb;pdb.set_trace()

from keras.callbacks import EarlyStopping
es_cb = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

import numpy as np
np.random.seed(40)

model_addCategory.fit(
    [
        X_train, depth_train, fault_train, shooter_train, server_train,
        point_train, game_train, sets_train
    ],
    y_train,
    batch_size=20,
    epochs=5,
    #callbacks=[es_cb],
    validation_data=([
        X_test, depth_test, fault_test, shooter_test, server_test, point_test,
        game_test, sets_test
    ], y_test))
