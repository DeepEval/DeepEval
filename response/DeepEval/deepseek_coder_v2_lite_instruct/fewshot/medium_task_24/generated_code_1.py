import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: <1x1 convolution, 3x3 convolution>
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1_1)
    branch1 = Dropout(0.25)(conv1_2)

    # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu')(conv2_2)
    conv2_4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv2_3)
    branch2 = Dropout(0.25)(conv2_4)

    # Branch 3: <max pooling>
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch3 = Dropout(0.25)(branch3)

    # Concatenate outputs from all branches
    merged = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flattened = Flatten()(merged)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model