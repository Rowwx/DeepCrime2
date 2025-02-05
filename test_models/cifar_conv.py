from __future__ import print_function
import tensorflow as tf
import os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

# loads and trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset, and evaluates the model's performance on a test set 17.01

def main(model_name):
    model_location = os.path.join('trained_models', model_name)
    
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # CIFAR-10 image dimensions
    img_rows, img_cols = 32, 32
    num_classes = 10
    
    # Check if the data format is channels_first or channels_last
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Check if model already exists
    if not os.path.exists(model_location):
        batch_size = 128 # how about 64 here?
        epochs = 3  # CIFAR-10 training usually requires more epochs (e.g., 10-50) 3 # 20.01 juan

        # Build the model
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

        # Compile the model
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),
                      metrics=['accuracy'])

        # Train the model
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                  verbose=1, validation_data=(x_test, y_test))
        
        # Save the trained model
        model.save(os.path.join('trained_models', 'cifar_trained.h5'))
        
        # Evaluate the model on the test set
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    else:
        # Load the trained model if it already exists
        model = tf.keras.models.load_model(model_location)
        
        # Recompile the model to ensure metrics are properly set
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),
                      metrics=['accuracy'])

        # Evaluate the model on the test set
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    return score

if __name__ == '__main__':
    score = main('')  # You can provide a model name like 'cifar10_trained.h5' if the model exists
