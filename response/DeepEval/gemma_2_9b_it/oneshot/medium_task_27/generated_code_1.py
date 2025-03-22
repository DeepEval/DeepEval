import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Branch 1 (3x3 convolutions)
    conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)

    # Branch 2 (5x5 convolutions)
    conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2_1)

    # Concatenate branches
    branch_output = Concatenate()([conv1_2, conv2_2])

    # Global average pooling
    gap = GlobalAveragePooling2D()(branch_output)

    # Fully connected layers with softmax for attention weights
    attn1 = Dense(units=128, activation='relu')(gap)
    attn2 = Dense(units=10, activation='softmax')(attn1)  

    # Weighted sum of branches
    weighted_output = keras.layers.multiply([conv1_2, attn2[:, :, 0]]) + keras.layers.multiply([conv2_2, attn2[:, :, 1]])

    # Final fully connected layer
    output_layer = Dense(units=10, activation='softmax')(weighted_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model