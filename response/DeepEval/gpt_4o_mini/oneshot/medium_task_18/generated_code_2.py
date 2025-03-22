import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, BatchNormalization

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32x3
    
    # First convolutional layer followed by max pooling
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block(input_tensor):
        # Different convolutional paths
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)  # 3x3 max pooling
        
        # Concatenating the outputs from different paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor

    # Building the block with the input from max pooling layer
    block_output = block(input_tensor=max_pooling)
    
    # Batch normalization after concatenation
    batch_norm = BatchNormalization()(block_output)
    
    # Flatten the output
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer for 10 classes (CIFAR-10)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model