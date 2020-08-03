import tensorflow as tf
import os
import numpy as np

from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense
from glob import glob

if not os.path.isdir('models'):
    os.mkdir('models')

print('TensorFlow version:', tf.__version__)
print('Is using GPU?', tf.test.is_gpu_available())

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, y_train = get_two_classes(x_train, y_train)
    x_test, y_test = get_two_classes(x_test, y_test)

    print(y_train.size)
    print(y_test.size)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    class_names = ['cats', 'dogs']
    show_random_examples(x_train, y_train, y_train, class_names)

    model = create_model()
    model.summary()

    h = model.fit(
        x_train / 255., y_train,
        validation_data=(x_test / 255., y_test),
        epochs=20, batch_size=128,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2),
            tf.keras.callbacks.ModelCheckpoint(
                'models/model_{val_accuracy:.3f}.h5',
                save_best_only=True, save_weights_only=False,
                monitor='val_accuracy'
            )
        ]
    )

    losses = h.history['loss']
    accs = h.history['accuracy']
    val_losses = h.history['val_loss']
    val_accs = h.history['val_accuracy']
    epochs = len(losses)

    plt.plot(range(len(accs)), accs, label='Training')
    plt.plot(range(len(accs)), val_accs, label='Validation')
    plt.legend()
    plt.show()


    saved_models = sorted(glob('models/model_*.h5')) # retrieve all paths of h5 models saved in this directory
    model = tf.keras.models.load_model(saved_models[-1]) # we load the last model as it will be the highest accuracy
    preds = model.predict(x_test / 255.)
    show_random_examples(x_test, y_test, preds, class_names)


def get_two_classes(x, y):
    """
    data preprocessing
    """
    # cats has y==3, while dogs y==5 acording to cifar10 dataset.
    indices_0, _ = np.where(y == 3.)
    indices_1, _ = np.where(y == 5.)

    indices = np.concatenate([indices_0, indices_1], axis=0)

    indices = np.random.choice(indices, indices.shape[0], replace=False)

    x = x[indices]
    y = y[indices]

    y = tf.keras.utils.to_categorical(y)[:, [3, 5]]

    return x, y

def show_random_examples(x, y, p, class_names):
    indicies = np.random.choice(range(x.shape[0]), 10, replace=False)

    x = x[indicies]
    y = y[indicies]
    p = p[indicies]

    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, 1 + i)
        plt.imshow(x[i])
        plt.xticks([])
        plt.yticks([])
        col = 'green' if np.argmax(y[i]) == np.argmax(p[i]) else 'red'
        plt.xlabel(class_names[np.argmax(p[i])], color=col)
    plt.show()

def create_model():
    def add_conv_block(model, num_filters):
        model.add(Conv2D(num_filters, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(num_filters, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.5))
        return model

    model = tf.keras.models.Sequential()
    model.add(Input(shape=(32, 32, 3)))

    model = add_conv_block(model, 32)
    model = add_conv_block(model, 64)
    model = add_conv_block(model, 128)

    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    main()