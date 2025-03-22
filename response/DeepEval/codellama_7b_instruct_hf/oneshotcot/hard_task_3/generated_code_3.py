import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # 1x1 convolution for each group
    conv_layer1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer)
    conv_layer2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(split_layer)
    conv_layer3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(split_layer)

    # Dropout layer for feature selection
    dropout_layer = Dropout(rate=0.2)(conv_layer1)

    # Concatenate outputs from three groups
    concat_layer = Concatenate()([conv_layer1, conv_layer2, conv_layer3])

    # 1x1 convolution to match output dimension of main pathway
    conv_layer4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer)

    # Addition operation for both pathways
    add_layer = Concatenate()([concat_layer, conv_layer4])

    # Batch normalization and flatten layer
    batch_norm = BatchNormalization()(add_layer)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model