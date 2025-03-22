import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Step 2: Add convolutional layers (for feature extraction)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)

    # Step 3: Add max pooling layers
    max_pool_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(conv2)
    max_pool_2x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)
    max_pool_4x4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(conv2)

    # Step 4: Flatten the outputs of each pooling layer
    flattened_1x1 = Flatten()(max_pool_1x1)
    flattened_2x2 = Flatten()(max_pool_2x2)
    flattened_4x4 = Flatten()(max_pool_4x4)

    # Step 5: Concatenate the flattened outputs
    concatenated = Concatenate()([flattened_1x1, flattened_2x2, flattened_4x4])

    # Step 6: Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Step 7: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model