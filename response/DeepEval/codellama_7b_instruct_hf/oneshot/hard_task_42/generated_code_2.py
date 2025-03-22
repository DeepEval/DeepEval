from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense


def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Max pooling layers with different scales
    block1_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1_conv1)
    block1_conv2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(block1_maxpool1)
    block1_maxpool2 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(block1_conv2)

    # Flatten and concatenate the outputs
    block1_flatten = Flatten()(block1_maxpool2)
    block1_dropout = Dropout(0.5)(block1_flatten)

    # Block 2: Convolutional and pooling layers with different features
    block2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_dropout)
    block2_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_dropout)
    block2_conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block1_dropout)
    block2_maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2_conv1)
    block2_maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2_conv2)
    block2_maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2_conv3)

    # Concatenate the outputs
    block2_concat = Concatenate()([block2_maxpool1, block2_maxpool2, block2_maxpool3])

    # Flatten and concatenate the outputs
    block2_flatten = Flatten()(block2_concat)
    block2_dropout = Dropout(0.5)(block2_flatten)

    # Fully connected layers
    fc1 = Dense(units=128, activation='relu')(block2_dropout)
    fc2 = Dense(units=64, activation='relu')(fc1)
    output_layer = Dense(units=10, activation='softmax')(fc2)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model