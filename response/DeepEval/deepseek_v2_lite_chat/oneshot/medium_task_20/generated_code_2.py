import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Average

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First path: 1x1 convolution, 3x3 convolution, 1x1 max pooling
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    path1 = BatchNormalization()(path1)
    path1 = Activation('relu')(path1)
    path1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(path1)
    path1 = MaxPooling2D(pool_size=(2, 2))(path1)

    # Second path: 1x1 convolution, 3x3 convolution
    path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    path2 = BatchNormalization()(path2)
    path2 = Activation('relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(path2)

    # Third path: 3x3 convolution, 1x1 max pooling
    path3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    path3 = BatchNormalization()(path3)
    path3 = Activation('relu')(path3)
    path3 = MaxPooling2D(pool_size=(2, 2))(path3)

    # Fourth path: 1x1 max pooling, 1x1 convolution
    path4 = MaxPooling2D(pool_size=(1, 1))(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(path4)

    # Concatenate the outputs of the four paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten and pass through dense layers
    flattened = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flattened)

    # Final dense layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Optional: print summary of the model
model.summary()