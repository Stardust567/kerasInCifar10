from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

nb_classes = 10
epochs=10
batch_size=128

def dataLoad():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # 重塑
    x_train_rows = x_train.reshape(x_train.shape[0], 32 * 32 * 3)
    x_test_rows = x_test.reshape(x_test.shape[0], 32 * 32 * 3)
    # 归一化
    x_train = minmax.fit_transform(x_train_rows)
    x_test = minmax.fit_transform(x_test_rows)
    # 重新变为32 x 32 x 3
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    # 对目标变量进行one-hot编码
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, y_train, x_test, y_test

def modelSet():
    # 构建模型
    model = Sequential()
    model.add(Convolution2D(32, (3, 3),
                            padding='same',
                            input_shape=(32, 32, 3)))  # 卷积层1
    model.add(Activation('relu'))  # 激活层
    model.add(Convolution2D(64, (3, 3)))  # 卷积层2
    model.add(Activation('relu'))  # 激活层
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 池化层
    #model.add(Dropout(0.25))  # 神经元随机失活
    model.add(Flatten())  # 拉成一维数据
    model.add(Dense(128))  # 全连接层1
    model.add(Activation('relu'))  # 激活层
    #model.add(Dropout(0.5))  # 随机失活
    model.add(Dense(nb_classes))  # 全连接层2
    model.add(Activation('softmax'))  # Softmax评分

    # 编译模型
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = dataLoad()
    model = modelSet()
    # 训练模型
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_test, y_test))
    # 保存模型
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save('model.h5')
    model.save_weights('model_weights.h5')
    # 评估模型
    score1 = model.evaluate(X_train, y_train, verbose=0)
    score2 = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score1[1], score2[1])
