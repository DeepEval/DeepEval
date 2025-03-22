import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block1(input_tensor):
        # Primary path
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path1)

        # Branch path
        path2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

        # Concatenate along the channel dimension
        output_tensor = Concatenate(axis=-1)([path1, path2])
        return output_tensor

    block1_output = block1(input_layer)
    batch_norm1 = BatchNormalization()(block1_output)
    flatten_layer = Flatten()(batch_norm1)

    # Block 2
    def block2(input_tensor):
        # Reshape the features from Block 1 into the target shape
        reshaped = Reshape((-1, 28, 28, 1))(input_tensor)
        
        # Permute the dimensions to swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)
        
        # Reshape back to the original shape to achieve channel shuffling
        shuffled = Reshape((28, 28, 1, -1))(permuted)
        
        return shuffled

    block2_output = block2(flatten_layer)
    batch_norm2 = BatchNormalization()(block2_output)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(batch_norm2)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage
model = dl_model()
model.summary()