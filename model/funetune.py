

import numpy as np
from getdata import load
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, concatenate,Dense, Dropout,TimeDistributed,RepeatVector, concatenate, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from keras import backend as k
from keras.layers import Dense, GlobalAveragePooling2D,Permute, Lambda
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
import tensorflow as tf
import warnings
import os
import lda
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
#from gettopic import get_train_test_lda
from keras.callbacks import TensorBoard
from keras import backend as K

warnings.filterwarnings("ignore")
import scipy.sparse as sp
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import accuracy_score,hamming_loss,average_precision_score,f1_score,recall_score, precision_score

def get_train_test_lda(topic):
    model = VGG16(include_top=False, pooling='avg')

    x_train, y_train, x_test, y_test = load()

    x_train = x_train.astype('float32')
    x_train /= 255

    y_train = y_train.astype('int64')

    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = y_test.astype('float32')

    X_train = model.predict(x_train)
    print(X_train.shape)
    X_test = model.predict(x_test)
    # X_train = model.predict(x_train)
    # X_test = model.predict(x_test)

    for k in topic:
        X_iter = X_train

        model_label = lda.LDA(n_topics=k, n_iter=1000)
        model_label.fit(y_train)
        doc_topic = model_label.doc_topic_
        x2 = doc_topic

        x = x2
        x = discretization_doc_topic(x)
        X_train = np.hstack((X_train, x))

        # multi-label learning to get x2
        classifier = LabelPowerset(RandomForestClassifier())
        classifier.fit(X_iter, x)

        x = np.array(sp.csr_matrix(classifier.predict(X_test)).toarray())
        # print(x)
        # x = alpha * x1 + (1-alpha) * x2
        # x = self.discretization_doc_topic(x)
        X_test = np.hstack((X_test, x))

    return np.array(X_train)[:,-28:], np.array(y_train), np.array(X_test)[:,-28:], np.array(y_test)


def discretization_doc_topic(theta):
    n, k = theta.shape
    Y = np.zeros((n, k))
    for i in range(n):
        MAX = np.max(theta[i])
        MIN = np.max(theta[i])
        for j in range(k):
            if (MAX - theta[i][j] < 1.0 / k):
                # if(theta[i][j] > MAX - 1.0/k):
                Y[i][j] = 1
            else:
                Y[i][j] = 0
    return Y


#get topic

#x_train_topic, y_train, x_val_topic, y_val = get_train_test_lda([2,3,5,7,11])
#np.savetxt("x_train_topic", x_train_topic)
#np.savetxt("x_val_topic", x_val_topic)

x_train_topic = np.loadtxt("x_train_topic")
x_val_topic = np.loadtxt("x_val_topic")
x_train, y_train, x_val, y_val = load()
#input1, input2 = get_train_test_lda(y_train, y_val, [7])
#train_topics, val_topics = get_train_test_lda()

x_train = x_train.astype('float32')
x_val = x_val.astype("float32")
#val_topics = val_topics.astype('float32')
#train_topics = train_topics.astype('float32')
#input1 = input1.astype('float32')
#input2 = input2.astype('float32')

x_train /= 255
x_val /= 255

print("model running...")
print("load VGG16 network...")
#base_model = VGG16(weights='imagenet', include_top=False)
base_model = ResNet50(weights='imagenet', include_top=False)
img_input = base_model.output
#img_input = Dropout(0.5)(img_input)

img_input = GlobalAveragePooling2D(name='globalaveragepool')(img_input)
#img_input = Dropout(0.5)(img_input)

#repeated_img_vector = RepeatVector(20)(img_input)

# base_inputs = Input(shape=(224,224,3))
# x = Convolution2D(32, kernel_size=(3, 3),padding='same',input_shape=(224, 224,3),data_format='channels_last')(base_inputs)
# x = Activation('relu')(x)
# x = Convolution2D(64, (3, 3))(x)
# x = Activation('relu')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x= Dropout(0.25)(x)
#
# x = Convolution2D(64,(3, 3), padding='same')(x)
# x = Activation('relu')(x)
# x = Convolution2D(64, 3, 3)(x)
# x = Activation('relu')(x)
# x = MaxPooling2D(pool_size=(2, 2))(x)
# x = Dropout(0.25)(x)
#
# x = Flatten()(x)
# x = Dense(512)(x)
# x = Activation('relu')(x)
# x = Dropout(0.5)(x)

auxiliary_input = Input(shape=(28,), name='aux_input')
#auxiliary_input.trainable = False
x = concatenate([img_input, auxiliary_input])

x = Dense(64, activation='relu')(x)

x = Dropout(0.5)(x)
predictions = Dense(20, activation='sigmoid')(x)
sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)

model = Model([base_model.input,auxiliary_input], outputs=predictions)

model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=['accuracy'])

#model.fit([x_train, train_topics], y_train, batch_size=32, epochs=3, callbacks=[TensorBoard(log_dir='./tmp/log')],validation_data=([x_val, val_topics], y_val))
#model.fit([x_train,x_train_topic], y_train, batch_size=32, epochs=30, callbacks=[TensorBoard(log_dir='./tmp/log')],validation_data=([x_val,x_val_topic], y_val))
#model.save_weights("weight.h5")
model.load_weights('weight.h5')
out = model.predict([x_val,x_val_topic])

out = np.array(out)
from mAP import get_AP
print(out)

AP = get_AP(y_val, out)
average = sum(AP) / 20
print(AP)
print(average)

np.savetxt("val_prediction.txt", out)

acc = []
accuracies = []
best_thershold= np.zeros(out.shape[1])

threshold = np.arange(0.1, 0.9, 0.1)

#best thershold for each binary classifier
for i in range(out.shape[1]):
    y_prob = np.array(out[:,i])

    for j  in threshold:
        y_pred = [1 if prob >=j else 0 for prob in y_prob]
        acc.append(matthews_corrcoef(y_val[:,i], y_pred))

    acc = np.array(acc)
    index = np.where(acc == acc.max())
    accuracies.append(acc.max())
    best_thershold[i] = threshold[index[0][0]]
    acc = []

print("best thresholds", best_thershold)
predictions = np.array([[1 if out[i,j] >= best_thershold[j] else 0 for j in range (y_val.shape[1])] for i in range(len(y_val))])
print(predictions)


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

