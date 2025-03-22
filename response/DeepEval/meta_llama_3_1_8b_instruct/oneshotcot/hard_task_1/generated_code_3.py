import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, MaxPooling2D, Multiply, Add, Activation
from keras import regularizers

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block1(input_tensor):
        # Path 1: Global average pooling followed by two fully connected layers
        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(units=128, activation='relu')(path1)
        path1 = Dense(units=10, activation='softmax')(path1)

        # Path 2: Global max pooling followed by two fully connected layers
        path2 = GlobalMaxPooling2D()(input_tensor)
        path2 = Dense(units=128, activation='relu')(path2)
        path2 = Dense(units=10, activation='softmax')(path2)

        # Add the outputs of both paths
        output_tensor = Add()([path1, path2])
        output_tensor = Activation('sigmoid')(output_tensor)

        # Apply channel attention weights to the original features
        output_tensor = Multiply()([output_tensor, input_tensor])

        return output_tensor
    
    block1_output = block1(conv)
    block2_input = block1_output

    def block2(input_tensor):
        # Extract spatial features by separately applying average pooling and max pooling
        path1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)

        # Concatenate the outputs along the channel dimension
        output_tensor = Concatenate()([path1, path2])

        # Apply a 1x1 convolution and sigmoid activation to normalize the features
        output_tensor = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(output_tensor)

        # Normalize the features
        output_tensor = Activation('sigmoid')(output_tensor)

        return output_tensor
        
    block2_output = block2(block1_output)
    block3_input = block2_output

    # Element-wise multiplication of Block 1 and Block 2 features
    block2_output = Multiply()([block1_output, block2_output])

    # Additional branch with a 1x1 convolutional layer
    branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block3_input)

    # Add the outputs of the main path and the additional branch
    output_tensor = Add()([block2_output, branch])
    output_tensor = Activation('relu')(output_tensor)

    # Final classification
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(output_tensor)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model