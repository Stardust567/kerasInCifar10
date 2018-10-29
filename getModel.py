from keras.models import Sequential
from keras.models import load_model
import cnn

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = cnn.dataLoad()
    model = load_model('model.h5')
    model.load_weights("model_weights.h5")
    score1 = model.evaluate(X_train, y_train, verbose=0)
    score2 = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score1[1], score2[1])