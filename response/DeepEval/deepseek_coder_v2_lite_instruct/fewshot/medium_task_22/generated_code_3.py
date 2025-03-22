import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First branch: 3x3 convolutions
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)

    # Second branch: 1x1 convolutions followed by two 3x3 convolutions
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_2)

    # Third branch: Max pooling
    pool = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)

    # Concatenate the outputs from the three branches
    fused_features = Concatenate()([conv1_2, conv2_3, pool])

    # Flatten the fused feature maps
    flattened_features = Flatten()(fused_features)

    # Pass the flattened output through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_features)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model