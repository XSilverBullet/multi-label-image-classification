

import numpy as np
from getdata import load
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, concatenate
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from keras import backend as k
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import warnings
import os
from sklearn.metrics import accuracy_score,hamming_loss,average_precision_score,f1_score,recall_score, precision_score
from keras.callbacks import TensorBoard
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


x_train, y_train, x_val , y_val = load()

#get topic
#input1, input2 = get_train_test_lda(y_train, y_val, [7])

x_train = x_train.astype('float32')
x_val = x_val.astype("float32")
#input1 = input1.astype('float32')
#input2 = input2.astype('float32')

x_train /= 255
x_val /= 255

print("model running...")
print("load VGG16 network...")
base_model = VGG16(weights='imagenet', include_top=False)
x = base_model.output

x = GlobalAveragePooling2D()(x)

# x = Dense(64, activation='relu')(x)
# x = Dropout(0.5)(x)

#x = Dense(32, activation='relu')(x)
#x = Dropout(0.6)(x)

#auxiliay_in = Input(shape=(7, ))
#x = concatenate([x, auxiliay_in])

#decoder_inputs = Input(shape=(None, 512))
# decoder_inputs = x
# # We set up our decoder to return full output sequences,
# # and to return internal states as well. We don't use the
# # return states in the training model, but we will use them in inference.
# decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
# #print(decoder_lstm)
# init = tf.constant(np.zeros((32,20,512)))
# decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
#                                      initial_state = init)
# decoder_dense = Dense(20, activation='sigmoid')
# decoder_outputs = decoder_dense(decoder_outputs)
x = Dense(64, activation='relu')(x)

x = Dropout(0.5)(x)

predictions = Dense(20, activation='sigmoid')(x)
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)


model = Model(base_model.input, outputs=predictions)


model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=30, callbacks=[TensorBoard(log_dir='./tmp/log')],validation_data=(x_val, y_val))

model.save_weights("weight_nolda.h5")
out = model.predict(x_val)
print(out)
out = np.array(out)

np.savetxt("val_prediction.txt", out)


from mAP import get_AP
print(out)

AP = get_AP(y_val, out)
average = sum(AP) / 20
print(AP)
print(average)

acc = []
accuracies = []
best_thershold= np.zeros(out.shape[1])

threshold = np.arange(0.1, 0.9, 0.1)


#best thershold for each binary classifier
for i in range(out.shape[1]):
    y_prob = np.array(out[:,i])

    for j in threshold:
        y_pred = [1 if prob >=j else 0 for prob in y_prob]
        acc.append(matthews_corrcoef(y_val[:,i], y_pred))

    acc = np.array(acc)
    index = np.where(acc == acc.max())
    accuracies.append(acc.max())
    best_thershold[i] = threshold[index[0][0]]
    acc = []


print("best thresholds", best_thershold)
predictions = np.array([[1 if out[i,j] >= best_thershold[j] else 0 for j in range (y_val.shape[1])] for i in range(len(y_val))])

print("-"*40)
print("Matthews Correlation Coefficient")
print("Class wise accuracies")
print(accuracies)

result1 = f1_score(y_val, predictions, average='micro')
print("f1-micro:",result1)
result2 = f1_score(y_val, predictions, average='macro')
print("f1-macro:", result2)

result3 = f1_score(y_val, predictions, average='samples')
print("f1-example:", result3)

result4 = hamming_loss(y_val, predictions)
print("hamming loss:", result4)

result5 = average_precision_score(y_val, predictions)
print("average_precision:", result5)

result6 = precision_score(y_val, predictions, average='micro')
print("precision:", result6)

result7 = recall_score(y_val, predictions, average='micro')
print("recall:", result7)

