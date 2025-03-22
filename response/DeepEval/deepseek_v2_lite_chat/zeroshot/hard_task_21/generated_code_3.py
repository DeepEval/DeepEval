import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Lambda, Add, Concatenate, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
input_shape = x_train[0].shape
num_classes = len(np.unique(y_train))


def main_path(input_shape):
    inputs = Input(shape=input_shape)
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    conv1_1 = Conv2D(32, (1, 1), padding='same')(split1[0])
    conv1_2 = Conv2D(32, (3, 3), padding='same')(split1[1])
    conv1_3 = Conv2D(32, (5, 5), padding='same')(split1[2])
    concat = Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])
    return Model(inputs=[inputs], outputs=[concat])


def branch_path(input_shape):
    output_shape = input_shape[0], input_shape[1], 64
    conv2 = Conv2D(64, (1, 1), padding='same')(input_shape)
    model = Model(inputs=inputs, outputs=conv2)
    return model


def dl_model(input_shape):
    main = main_path(input_shape)
    branch = branch_path(input_shape)
    main_output = main(inputs)
    branch_output = branch(inputs)
    combined_output = Add()([main_output, branch_output])
    fully_connected = Dense(10, activation='softmax')(combined_output)
    model = Model(inputs=[main.input, branch.input], outputs=fully_connected)
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=[CategoricalAccuracy()])
    return model


model = dl_model((32, 32, 3))
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))