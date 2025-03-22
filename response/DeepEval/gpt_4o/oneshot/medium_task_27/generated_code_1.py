import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: Convolution with 3x3 kernel
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: Convolution with 5x5 kernel
    conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine branches through addition
    combined = Add()([conv3x3, conv5x5])

    # Global average pooling
    gap = GlobalAveragePooling2D()(combined)

    # Fully connected layers to generate attention weights
    attention_dense1 = Dense(units=64, activation='relu')(gap)
    attention_weights = Dense(units=2, activation='softmax')(attention_dense1)

    # Split attention weights for each branch
    weight1 = Multiply()([conv3x3, attention_weights[:, 0:1]])
    weight2 = Multiply()([conv5x5, attention_weights[:, 1:2]])

    # Combine weighted branch outputs
    weighted_output = Add()([weight1, weight2])

    # Flatten the weighted output
    final_gap = GlobalAveragePooling2D()(weighted_output)

    # Fully connected layer for final classification
    output_layer = Dense(units=10, activation='softmax')(final_gap)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model