import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Concatenate, DepthwiseConv2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Average Pooling with different scales
    def block_1(input_tensor):
        # Path 1: 1x1 average pooling
        avgpool1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        # Path 2: 2x2 average pooling
        avgpool2x2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        # Path 3: 4x4 average pooling
        avgpool4x4 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        # Flatten and concatenate
        flatten = Flatten()(Concatenate()([avgpool1x1, avgpool2x2, avgpool4x4]))
        return flatten

    # Block 2: Depthwise Separable Convolutions
    def block_2(input_tensor):
        # Path 1: 1x1 convolution, 3x3 convolution, 1x7 convolution, 7x1 convolution
        conv1x1_3x3_1x7_7x1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x1, conv3x3, conv1x7, conv7x1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Average pooling
        avgpool1x1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        avgpool3x3 = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(input_tensor)
        avgpool7x7 = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(input_tensor)
        # Concatenate and flatten
        concat = Concatenate()([conv1x1_3x3_1x7_7x1, conv1x1, conv3x3, conv1x7, conv7x1, avgpool1x1, avgpool3x3, avgpool7x7])
        flatten = Flatten()(concat)
        return flatten

    # Processing blocks
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=input_layer)

    # Concatenate and reshape for fully connected layers
    concatenated = Concatenate()([block1_output, block2_output])
    reshaped = Reshape((-1, 16))(concatenated)  # Adjust the output size for the fully connected layers

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(reshaped)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])