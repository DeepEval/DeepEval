import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():     
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # Step 2: Define branches
    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3: 1x1 Convolution followed by 5x5 Convolution
    branch3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Branch 4: 3x3 Max Pooling followed by 1x1 Convolution
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

    # Step 4: Concatenate the outputs of the branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # Step 5: Flatten the result
    flatten_layer = Flatten()(concatenated)

    # Step 6: Add two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Step 7: Output layer for classification (10 classes for CIFAR-10)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Step 8: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model