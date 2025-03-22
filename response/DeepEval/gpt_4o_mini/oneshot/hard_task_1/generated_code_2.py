import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, Concatenate, BatchNormalization

def dl_model():
    # Input layer for CIFAR-10 images (32x32 pixels, 3 color channels)
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to adjust the number of output channels
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1: Channel attention mechanism
    def channel_attention_block(input_tensor):
        # Path 1: Global Average Pooling followed by two fully connected layers
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1_avg = Dense(units=32, activation='relu')(avg_pool)
        dense2_avg = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1_avg)

        # Path 2: Global Max Pooling followed by two fully connected layers
        max_pool = GlobalMaxPooling2D()(input_tensor)
        dense1_max = Dense(units=32, activation='relu')(max_pool)
        dense2_max = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1_max)

        # Combine both paths by adding
        channel_attention = Add()([dense2_avg, dense2_max])
        channel_attention = Activation('sigmoid')(channel_attention)

        # Scale input features by channel attention weights
        scale = Multiply()([input_tensor, channel_attention])

        return scale

    # Apply the channel attention block
    block1_output = channel_attention_block(conv1)

    # Block 2: Spatial feature extraction
    avg_pool = GlobalAveragePooling2D()(block1_output)
    max_pool = GlobalMaxPooling2D()(block1_output)
    concat_output = Concatenate()([avg_pool, max_pool])
    
    # 1x1 convolution to normalize features
    conv2 = Conv2D(filters=block1_output.shape[-1], kernel_size=(1, 1), padding='same', activation='sigmoid')(concat_output)
    
    # Multiply with Block 1 output features
    final_features = Multiply()([block1_output, conv2])
    
    # Additional branch with a 1x1 convolution to align channels
    conv3 = Conv2D(filters=block1_output.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(final_features)
    
    # Add to the main path
    added_output = Add()([final_features, conv3])
    activated_output = Activation('relu')(added_output)

    # Fully connected layer for classification
    flatten_layer = keras.layers.Flatten()(activated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model