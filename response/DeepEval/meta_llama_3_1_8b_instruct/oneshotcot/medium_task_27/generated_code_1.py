import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Multiply, Lambda

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs of the two convolutional branches
    branch_output = Add()([conv3x3, conv5x5])

    # Apply global average pooling to compress the features
    global_avg_pooling = GlobalAveragePooling2D()(branch_output)

    # Apply fully connected layers to generate attention weights
    dense1 = Dense(units=128, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Generate the weighted output
    weighted_output = Multiply()([dense2, dense2])  # element-wise multiplication

    # Apply a fully connected layer to deliver the probability distribution
    output_layer = Dense(units=10, activation='softmax')(weighted_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model