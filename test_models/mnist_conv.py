from __future__ import print_function
import tensorflow as tf
import os
from tensorflow.keras.datasets import mnist  # Corrected import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

# loads and trains a Convolutional Neural Network (CNN) on the MNIST dataset, and evaluates the model's performance on a test set 17.01.2025 juan


def main(model_name):
    model_location = os.path.join('trained_models', model_name)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # Corrected load_data()

    img_rows, img_cols = 28, 28
    num_classes = 10

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    if not os.path.exists(model_location):
        batch_size = 128
        epochs = 3  # 20.12 juan

        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                  verbose=1, validation_data=(x_test, y_test))
        model.save(os.path.join('trained_models', 'mnist_trained.h5'))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    else:
        model = tf.keras.models.load_model(model_location)
        # Recompile the model to ensure metrics are properly set
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),
                      metrics=['accuracy'])

        score = model.evaluate(x_test, y_test, verbose=0)
        print('score:', score)

    return score


if __name__ == '__main__':
    score = main('')