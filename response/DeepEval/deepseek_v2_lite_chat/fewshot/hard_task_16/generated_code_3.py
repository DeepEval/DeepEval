import keras
from keras.layers import Input, Lambda, Concatenate, Conv2D, MaxPool2D, Flatten, Dense, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Splitting and Convolution
    def block_1(input_tensor):
        # Split input tensor into three groups
        split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        # Convolution operations
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_tensors[0])
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(split_tensors[1])
        conv1_3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_tensors[2])
        # Concatenate the outputs from the three groups
        concatenated = Concatenate()(inputs=[conv1_1, conv1_2, conv1_3])
        return concatenated

    # Block 2: Global Max Pooling
    def block_2(input_tensor):
        # Global max pooling
        maxpool = MaxPool2D(pool_size=(4, 4))(input_tensor)
        # Fully connected layers for generating weights
        fc1 = Dense(units=512, activation='relu')(maxpool)
        fc2 = Dense(units=512, activation='relu')(fc1)
        # Reshape the weights to match the shape of the adjusted output
        weights = Reshape((-1, fc1.shape[-1]))(fc2)
        # Multiply the reshaped weights with the adjusted output
        multiplied = keras.layers.dot([input_tensor, weights], axes=-1)
        return multiplied

    # Block 1 output
    block1_output = block_1(input_layer)
    # Block 2 output
    block2_output = block_2(block1_output)
    # Concatenate outputs from both blocks
    concatenated_output = Concatenate()([block1_output, block2_output])
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(concatenated_output)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and return the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
return model