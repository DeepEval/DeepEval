import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Flatten, concatenate, Conv2DTranspose

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Convolutional layer
    x = Conv2D(32, (3, 3), activation='relu')(inputs)

    # Batch normalization and ReLU activation
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Two fully connected layers
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Reshape output to match initial feature maps
    output = Flatten()(x)
    output = Reshape((32, 32, 1))(output)

    # Multiply output with initial features to generate weighted feature maps
    output = Multiply()([output, inputs])

    # Concatenate output with input layer
    output = concatenate([output, inputs], axis=1)

    # Reduce dimensionality and downsample feature using 1x1 convolution and average pooling
    output = Conv2DTranspose(16, (3, 3), strides=2, padding='same')(output)
    output = GlobalAveragePooling2D()(output)

    # Final fully connected layer for classification
    output = Dense(10, activation='softmax')(output)

    model = keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model