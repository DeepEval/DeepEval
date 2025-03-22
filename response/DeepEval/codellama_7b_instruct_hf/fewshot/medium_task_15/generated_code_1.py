import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Flatten, Dense, Reshape, Multiply


def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv_layer = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Batch normalization layer
    batch_norm_layer = BatchNormalization()(conv_layer)

    # ReLU activation layer
    relu_layer = ReLU()(batch_norm_layer)

    # Global average pooling layer
    gap_layer = GlobalAveragePooling2D()(relu_layer)

    # Flatten layer
    flatten_layer = Flatten()(gap_layer)

    # Fully connected layers
    fc_layer1 = Dense(units=64, activation='relu')(flatten_layer)
    fc_layer2 = Dense(units=10, activation='softmax')(fc_layer1)

    # Reshape layer
    reshaped_layer = Reshape(target_shape=(32, 32, 3))(fc_layer2)

    # Multiply layer
    multiplied_layer = Multiply()([input_layer, reshaped_layer])

    # Concatenate layers
    concatenated_layer = Concatenate()([input_layer, multiplied_layer])

    # 1x1 convolution layer
    conv_layer2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_layer)

    # Average pooling layer
    average_pooling_layer = GlobalAveragePooling2D()(conv_layer2)

    # Flatten layer
    flatten_layer2 = Flatten()(average_pooling_layer)

    # Fully connected layer
    fc_layer3 = Dense(units=64, activation='relu')(flatten_layer2)
    fc_layer4 = Dense(units=10, activation='softmax')(fc_layer3)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=fc_layer4)

    return model