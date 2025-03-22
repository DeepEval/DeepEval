import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First block with four parallel branches
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    
    # Concatenate the outputs of the parallel branches
    block1_output = Concatenate()([path1, path2, path3, path4])

    # Second block with global average pooling
    global_avg_pooling = GlobalAveragePooling2D()(block1_output)

    # Fully connected layers after global average pooling
    dense1 = Dense(units=128, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Generate weights that match the channels of block 2's input
    weights = Dense(units=block1_output.shape[-1], activation='sigmoid')(dense2)
    
    # Reshape weights to match input feature map's shape
    reshaped_weights = Reshape((1, 1, block1_output.shape[-1]))(weights)

    # Multiply the reshaped weights with the original feature map
    weighted_feature_map = Multiply()([block1_output, reshaped_weights])

    # Final fully connected layer to produce the output probability distribution
    flatten_layer = Flatten()(weighted_feature_map)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()