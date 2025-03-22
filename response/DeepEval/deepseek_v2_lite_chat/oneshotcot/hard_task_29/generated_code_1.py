import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Main path with dimension increase
    def block1(input_tensor):
        # Convolution layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        conv1_bn = BatchNormalization()(conv1)
        conv1_relu = keras.activations.relu(conv1_bn)

        # Channel restore convolution layer
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv1_relu)
        conv2_bn = BatchNormalization()(conv2)
        conv2_relu = keras.activations.relu(conv2_bn)

        # Branch path
        branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)

        # Combine outputs
        output1 = conv2_relu
        output2 = branch
        combined = Concatenate()([output1, output2])

        return combined

    block1_output = block1(input_tensor=input_layer)

    # Block 2: Variable pooling with dense paths
    def block2(input_tensor):
        # Pooling layers with different window sizes and strides
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_tensor)

        # Flatten and concatenate
        flat = Flatten()(pool3)
        concatenated = Concatenate()([flat, pool2, pool1])

        # Dense layers
        dense1 = Dense(units=128, activation='relu')(concatenated)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)

        return output_layer

    block2_output = block2(input_tensor=block1_output)

    # Final model
    model = keras.Model(inputs=input_layer, outputs=block2_output)

    return model

# Create and display the model
model = dl_model()
model.summary()