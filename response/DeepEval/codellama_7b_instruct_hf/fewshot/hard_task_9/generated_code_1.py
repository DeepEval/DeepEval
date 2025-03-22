import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Feature extraction branch 1: 1x1 convolution
    branch1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Feature extraction branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Feature extraction branch 3: 1x1 convolution followed by 2x2 convolution
    branch3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Concatenate outputs from all feature extraction branches
    merged = Concatenate()([branch1, branch2, branch3])

    # Adjust output dimensions to match input image's channel size
    merged = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(merged)

    # Add the feature extraction branches and the main path
    added = Add()([merged, input_layer])

    # Pass the output through a batch normalization layer
    batch_norm = BatchNormalization()(added)

    # Flatten the output and pass it through a fully connected layer
    flattened = Flatten()(batch_norm)
    fc1 = Dense(units=128, activation='relu')(flattened)
    fc2 = Dense(units=64, activation='relu')(fc1)
    output_layer = Dense(units=10, activation='softmax')(fc2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model