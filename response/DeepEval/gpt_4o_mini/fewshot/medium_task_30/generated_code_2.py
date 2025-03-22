import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # Block to apply average pooling with different scales
    def pooling_block(input_tensor):
        # Average pooling with 1x1 kernel
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)
        
        # Average pooling with 2x2 kernel
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(pool2)
        
        # Average pooling with 4x4 kernel
        pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(pool3)

        # Concatenate the flattened outputs
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    # Get the output of the pooling block
    pooled_output = pooling_block(input_layer)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(pooled_output)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer for classification (10 classes for CIFAR-10)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model