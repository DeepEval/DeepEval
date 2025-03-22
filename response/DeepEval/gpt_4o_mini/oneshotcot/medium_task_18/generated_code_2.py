import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, BatchNormalization

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are of shape 32x32 with 3 color channels (RGB)

    # Step 2: Add convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Add maxpooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    # Step 4: Define a block
    def block(input_tensor):
        # Step 4.1: 1x1 Convolution path
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Step 4.2: 3x3 Convolution path
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Step 4.3: 5x5 Convolution path
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Step 4.4: 3x3 MaxPooling path
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Step 4.5: Concatenate the outputs of the paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor
    
    # Step 5: Get the block output
    block_output = block(input_tensor=max_pooling)
    
    # Step 6: Add batch normalization layer
    batch_norm = BatchNormalization()(block_output)
    
    # Step 7: Add flatten layer
    flatten_layer = Flatten()(batch_norm)
    
    # Step 8: Add first dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 9: Add second dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 10: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model