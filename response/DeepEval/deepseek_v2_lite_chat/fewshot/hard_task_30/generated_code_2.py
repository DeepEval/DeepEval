import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Dual-path structure
    def block1(input_tensor):
        def conv_block(filters, kernel_size, strides, padding, activation):
            return Conv2D(filters, kernel_size, strides, padding, activation=activation)(input_tensor)

        # Main path
        conv1 = conv_block(32, (3, 3), (1, 1), 'same', 'relu')
        conv2 = conv_block(64, (3, 3), (1, 1), 'same', 'relu')
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # Branch path
        branch = conv_block(64, (3, 3), (1, 1), 'same', 'relu')(input_tensor)

        # Add main path and branch path
        add_layer = Add()([pool1, branch])

        # Flatten and fully connected layer
        flatten_layer = Flatten()(add_layer)
        dense1 = Dense(128, activation='relu')(flatten_layer)

        return dense1

    # Block 2: Split input into groups and process with depthwise separable convolutions
    def block2(input_tensor):
        # Split input into three groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

        # Process each group with a depthwise separable convolutional layer
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])

        # Concatenate the outputs of the three groups
        concat_layer = Concatenate()([conv1, conv2, conv3])

        # Flatten and fully connected layers
        flatten_layer = Flatten()(concat_layer)
        dense1 = Dense(128, activation='relu')(flatten_layer)
        dense2 = Dense(64, activation='relu')(dense1)
        output_layer = Dense(10, activation='softmax')(dense2)

        return output_layer

    # Model architecture
    model = Model(inputs=input_layer, outputs=[block1(input_tensor=input_layer), block2(input_tensor=input_layer)])

    return model

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()