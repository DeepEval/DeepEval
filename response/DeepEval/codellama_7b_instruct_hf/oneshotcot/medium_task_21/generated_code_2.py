import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

    # Concatenate the outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # Apply batch normalization and flatten the result
    batch_norm = BatchNormalization()(concatenated)
    flattened = Flatten()(batch_norm)

    # Apply dropout layers to mitigate overfitting
    dropout1 = Dropout(rate=0.2)(flattened)
    dropout2 = Dropout(rate=0.2)(dropout1)
    dropout3 = Dropout(rate=0.2)(dropout2)

    # Pass the output through three fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(dropout3)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model