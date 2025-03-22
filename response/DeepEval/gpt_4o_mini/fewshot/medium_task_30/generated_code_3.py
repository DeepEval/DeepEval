import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # Block with average pooling layers
    def pooling_block(input_tensor):
        avgpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(avgpool1)
        
        avgpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(avgpool2)
        
        avgpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(avgpool3)
        
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    # Process the input through the pooling block
    pooled_output = pooling_block(input_layer)

    # Further flatten the concatenated features
    flatten_output = Flatten()(pooled_output)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_output)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Final output layer with softmax activation for 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model