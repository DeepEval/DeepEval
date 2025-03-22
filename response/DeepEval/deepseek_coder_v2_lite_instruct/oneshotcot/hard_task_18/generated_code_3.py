import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Multiply, Flatten
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(input_tensor):
        # First 3x3 convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        # Second 3x3 convolutional layer
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        # Average pooling layer
        avg_pool = AveragePooling2D(pool_size=(2, 2))(conv2)
        # Add the input to the output of the average pooling
        added = Add()([input_tensor, avg_pool])
        return added

    # Apply the first block to the input layer
    first_block_output = first_block(input_layer)

    # Second block
    def second_block(input_tensor):
        # Global average pooling to generate channel weights
        gap = GlobalAveragePooling2D()(input_tensor)
        # Fully connected layer to refine weights
        fc1 = Dense(units=32, activation='relu')(gap)
        # Another fully connected layer
        fc2 = Dense(units=32, activation='relu')(fc1)
        # Reshape the output to match the input shape
        weights = Dense(units=input_tensor.shape[-1], activation='sigmoid')(fc2)
        # Multiply the input by the learned weights
        weighted_input = Multiply()([input_tensor, weights])
        return weighted_input

    # Apply the second block to the first block's output
    second_block_output = second_block(first_block_output)

    # Flatten the output
    flattened = Flatten()(second_block_output)

    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model