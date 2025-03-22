import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Multiply, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block with four parallel paths
    def first_block(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor

    # Apply the first block
    block1_output = first_block(input_layer)

    # Second block with global average pooling and attention-like mechanism
    def second_block(input_tensor):
        # Global average pooling
        pooled_features = GlobalAveragePooling2D()(input_tensor)

        # Fully connected layers to generate weights
        dense1 = Dense(units=64, activation='relu')(pooled_features)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)

        # Reshape weights to match the input's shape and apply them
        weights = Reshape((1, 1, input_tensor.shape[-1]))(dense2)
        weighted_features = Multiply()([input_tensor, weights])

        return weighted_features

    # Apply the second block
    block2_output = second_block(block1_output)

    # Final fully connected layer to produce the output probability distribution
    flatten_layer = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the Keras Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model