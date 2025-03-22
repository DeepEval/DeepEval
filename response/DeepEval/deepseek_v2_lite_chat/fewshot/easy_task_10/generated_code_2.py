import keras
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Concatenate, AveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='same')(input_layer)
    # 3x3 depthwise separable convolutional layer
    dw_conv1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    # 1x1 convolutional layer to reduce dimensionality
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv1)
    # Stride 2 applied to all convolutional layers
    avg_pool = AveragePooling2D(pool_size=(2, 2))(conv2)

    # Concatenate outputs from different layers
    concat = Concatenate()([avg_pool, conv1])

    # Flatten and fully connected layer
    flatten = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()