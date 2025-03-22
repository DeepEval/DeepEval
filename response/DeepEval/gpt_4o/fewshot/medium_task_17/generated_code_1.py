import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Reshape, Permute, Flatten

def dl_model():
    # CIFAR-10 images are 32x32 with 3 channels
    input_layer = Input(shape=(32, 32, 3))

    # Reshape the input to (height, width, groups, channels_per_group)
    def reshape_for_shuffle(x):
        batch_size, height, width, channels = tf.unstack(tf.shape(x))
        channels_per_group = channels // 3
        reshaped = tf.reshape(x, (batch_size, height, width, 3, channels_per_group))
        return reshaped
    
    reshaped = Lambda(reshape_for_shuffle)(input_layer)

    # Permute the dimensions to enable channel shuffling
    permuted = Permute((1, 2, 4, 3))(reshaped)

    # Reshape back to original shape (32, 32, 3)
    reshaped_back = Reshape((32, 32, 3))(permuted)

    # Flatten and pass through a fully connected layer for classification
    flatten_layer = Flatten()(reshaped_back)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model