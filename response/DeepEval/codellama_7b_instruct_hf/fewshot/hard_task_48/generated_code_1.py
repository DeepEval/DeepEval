import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Block 1
    inputs = Input(shape=(32, 32, 3))
    branch1 = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(inputs)
    branch1_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1_conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1_concat = Concatenate()([branch1_conv1, branch1_conv2, branch1_conv3])
    branch1_batch_norm = BatchNormalization()(branch1_concat)
    branch1_flatten = Flatten()(branch1_batch_norm)

    # Block 2
    branch2 = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(inputs)
    branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch2_conv1)
    branch2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool1)
    branch2_conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch2_conv2)
    branch2_concat = Concatenate()([branch2_conv1, branch2_conv2, branch2_conv3])
    branch2_batch_norm = BatchNormalization()(branch2_concat)
    branch2_flatten = Flatten()(branch2_batch_norm)

    # Final classification
    output_layer = Dense(units=10, activation='softmax')(Concatenate()([branch1_flatten, branch2_flatten]))

    model = Model(inputs=inputs, outputs=output_layer)

    return model