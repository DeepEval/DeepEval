import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer with 1x1 convolution to increase dimensionality
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 2), padding='same', activation='relu')(input_layer)

    # Depthwise separable 3x3 convolution for feature extraction
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Reduce dimensionality with 1x1 convolutional layer with stride 2
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 2), padding='same', activation='relu')(conv2)

    # Pooling layer with a pool size of 2
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv3)

    # Concatenate the outputs from the four parallel paths
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block_output = block(pool)
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)

    # Fully connected layer with 128 units
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Another fully connected layer with 64 units
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer with 10 units and softmax activation for classification probabilities
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.summary()