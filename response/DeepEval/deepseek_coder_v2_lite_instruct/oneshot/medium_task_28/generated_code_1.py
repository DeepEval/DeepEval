import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Multiply, Lambda
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Attention mechanism
    attention = Conv2D(1, (1, 1), activation='sigmoid')(input_layer)
    attention_weighted_input = Multiply()([input_layer, attention])

    # First 1x1 convolution to reduce dimensions
    reduced = Conv2D(32, (1, 1), padding='same')(attention_weighted_input)
    reduced = BatchNormalization()(reduced)
    reduced = tf.nn.relu(reduced)

    # Second 1x1 convolution to restore dimensions
    restored = Conv2D(32, (1, 1), padding='same')(reduced)
    restored = BatchNormalization()(restored)
    restored = tf.nn.relu(restored)

    # Add processed output to the original input
    output = Add()([input_layer, restored])

    # Flatten and fully connected layer
    flattened = Flatten()(output)
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model