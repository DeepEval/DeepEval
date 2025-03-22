import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Lambda layer to split input into three groups
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Feature extraction with depthwise separable convolutions
    path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])

    # Concatenate the outputs of the three paths
    concatenated = Concatenate()([path1, path2, path3])

    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(concatenated)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model