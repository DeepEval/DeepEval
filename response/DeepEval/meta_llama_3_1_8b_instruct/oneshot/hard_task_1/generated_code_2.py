import keras
from keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, Concatenate, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 image size is 32x32x3

    # Convolutional layer
    conv = Conv2D(filters=input_layer.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block1(input_tensor):
        # Path1: Global average pooling followed by two fully connected layers
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(avg_pool)
        dense2 = Dense(units=128, activation='relu')(dense1)

        # Path2: Global max pooling followed by two fully connected layers
        max_pool = GlobalMaxPooling2D()(input_tensor)
        dense3 = Dense(units=128, activation='relu')(max_pool)
        dense4 = Dense(units=128, activation='relu')(dense3)

        # Add and apply activation function to generate channel attention weights
        add_layer = Add()([dense2, dense4])
        activation_layer = Dense(units=input_tensor.shape[-1], activation='sigmoid')(add_layer)

        # Multiply element-wise with the original features
        multiply_layer = Multiply()([input_tensor, activation_layer])

        return multiply_layer

    block1_output = block1(conv)

    def block2(input_tensor):
        # Average pooling and max pooling separately
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)

        # Concatenate along the channel dimension
        concat_layer = Concatenate()([avg_pool, max_pool])

        # Apply 1x1 convolution and sigmoid activation to normalize features
        conv1x1 = Conv2D(filters=input_tensor.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(concat_layer)

        # Multiply element-wise with the channel dimension features from Block 1
        multiply_layer = Multiply()([conv1x1, block1_output])

        return multiply_layer

    block2_output = block2(block1_output)

    # Additional branch to align output channels with input channels
    add_layer = Add()([block2_output, input_tensor])

    # Final convolutional layer
    conv2 = Conv2D(filters=add_layer.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(add_layer)

    # Final classification through fully connected layer
    output_layer = Dense(units=10, activation='softmax')(conv2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model