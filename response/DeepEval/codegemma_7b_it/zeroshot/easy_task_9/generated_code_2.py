from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, DepthwiseConv2D, ZeroPadding2D, Input, Add, Flatten, Dense
from tensorflow.keras import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # 1x1 convolutional layer to increase dimensionality
    x = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 3x3 depthwise separable convolutional layer for feature extraction
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 1x1 convolutional layer to reduce dimensionality, maintaining a convolutional stride of 1 throughout
    x = Conv2D(filters=16, kernel_size=(1, 1), padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Add the original input to the processed output
    outputs = Add()([inputs, x])

    # Flatten and fully connected layer for classification
    outputs = Flatten()(outputs)
    outputs = Dense(10, activation='softmax')(outputs)

    # Create and return the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Example usage:
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train)