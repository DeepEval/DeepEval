import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Split the input into three groups along the last dimension
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Feature extraction using depthwise separable convolutions
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split[2])

    # Concatenate the outputs of the three paths
    concatenated = Concatenate()([path1, path2, path3])

    # Flatten the concatenated features
    flatten_layer = Flatten()(concatenated)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model