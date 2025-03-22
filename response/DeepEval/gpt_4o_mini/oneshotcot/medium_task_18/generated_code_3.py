import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():     
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels (RGB)

    # Step 2: Add convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Add max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    # Step 4: Define a block
    def block(input_tensor):
        # Step 4.1: Add 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Step 4.2: Add 3x3 convolution
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Step 4.3: Add 5x5 convolution
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Step 4.4: Add 3x3 max pooling
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Step 4.5: Add concatenate layer to merge the above paths
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor
    
    # Apply the block to the max pooling output
    block_output = block(input_tensor=max_pooling)
    
    # Step 5: Add batch normalization layer
    batch_norm = BatchNormalization()(block_output)
    
    # Step 6: Add flatten layer
    flatten_layer = Flatten()(batch_norm)
    
    # Step 7: Add first dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 8: Add second dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 9: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model