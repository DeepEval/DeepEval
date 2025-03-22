import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 2: Convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)

    # Step 4: Define the block
    def block(input_tensor):
        # Step 4.1: 1x1 convolution path
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Step 4.2: 3x3 convolution path
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Step 4.3: 5x5 convolution path
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Step 4.4: 3x3 max pooling path
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Step 4.5: Concatenate the paths
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor
    
    # Apply the block to the max pooling output
    block_output = block(input_tensor=max_pooling)

    # Step 5: Batch normalization layer
    batch_norm = BatchNormalization()(block_output)

    # Step 6: Flatten layer
    flatten_layer = Flatten()(batch_norm)

    # Step 7: First dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Step 8: Second dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Step 9: Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model