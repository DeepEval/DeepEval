import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, Dense, Reshape, DepthwiseConv2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Average pooling with different scales
    def block_1(input_tensor):
        # Average pooling with 1x1, 2x2, and 4x4 scales
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        # Flatten and concatenate the results
        flat1 = Flatten()(maxpool1)
        flat2 = Flatten()(maxpool2)
        flat3 = Flatten()(maxpool3)
        concat = Concatenate()([flat1, flat2, flat3])
        return concat

    # Block 2: Depthwise separable convolutions
    def block_2(input_tensor):
        # Split input into four groups for depthwise separable convolutions
        split = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(input_tensor)
        # Depthwise separable convolutions with different kernel sizes
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split[2])
        conv4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split[3])
        # Concatenate the outputs
        concat = Concatenate(axis=-1)([conv1, conv2, conv3, conv4])
        return concat

    # Process the input through both blocks
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    # Reshape the output to prepare for the final dense layers
    reshape = Reshape(target_shape=(16,))(block2_output)  # Adjust this based on the actual output dimensions
    dense = Dense(units=128, activation='relu')(reshape)
    output = Dense(units=10, activation='softmax')(dense)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()