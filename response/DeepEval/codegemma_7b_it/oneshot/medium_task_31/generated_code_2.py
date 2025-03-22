import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input along the channel dimension
    group1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3, num=0))(input_layer)
    group2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3, num=1))(input_layer)
    group3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3, num=2))(input_layer)

    # Apply convolutional kernels to each group
    conv1_group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
    conv1_group2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group2)
    conv1_group3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group3)

    conv2_group1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_group1)
    conv2_group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_group2)
    conv2_group3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_group3)

    conv3_group1 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2_group1)
    conv3_group2 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2_group2)
    conv3_group3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2_group3)

    # Concatenate the outputs from all groups
    concat = Concatenate()([conv3_group1, conv3_group2, conv3_group3])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model