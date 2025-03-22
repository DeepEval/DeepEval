import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Multiply, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block with four parallel branches
    def first_block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    first_block_output = first_block(input_layer)

    # Second block that reduces dimensionality using global average pooling
    def second_block(input_tensor):
        pooled_features = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=pooled_features.shape[-1], activation='relu')(pooled_features)
        dense2 = Dense(units=pooled_features.shape[-1], activation='sigmoid')(dense1)
        
        # Reshape weights and multiply with the input feature map
        weights = Reshape((1, 1, input_tensor.shape[-1]))(dense2)
        scaled_features = Multiply()([input_tensor, weights])
        return scaled_features

    second_block_output = second_block(first_block_output)

    # Final fully connected layer to produce the output
    output_layer = GlobalAveragePooling2D()(second_block_output)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model