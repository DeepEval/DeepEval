import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Lambda, Add, Flatten, Concatenate, Reshape, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block_1(input_tensor):
        # Path 1: 1x1 convolution
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Path 2: 3x3 depthwise separable convolution
        conv1_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Concatenation
        concat1 = Concatenate(axis=-1)([conv1_1, conv1_2])
        return concat1

    # Block 2
    def block_2(input_tensor):
        # Calculate the shape of features from Block 1
        shape_tensor = Lambda(lambda x: tf.shape(x), output_shape=(1, ))[0](input_tensor)
        height, width, channels = shape_tensor
        # Split into four groups
        groups = Lambda(lambda x: tf.split(x, 4, axis=-1))(input_tensor)
        # Process each group
        conv2_1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
        conv2_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(groups[1])
        conv2_3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(groups[2])
        conv2_4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(groups[3])
        # Concatenate along channel axis
        concat2 = Concatenate(axis=-1)([conv2_1, conv2_2, conv2_3, conv2_4])
        # Reshape and output layer
        reshaped = Reshape((height * width, channels * 4))(concat2)
        dense = Dense(units=128, activation='relu')(reshaped)
        output_layer = Dense(units=10, activation='softmax')(dense)
        return output_layer

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=block_2(block_1(input_layer)))
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Optional: Display the model summary
model.summary()