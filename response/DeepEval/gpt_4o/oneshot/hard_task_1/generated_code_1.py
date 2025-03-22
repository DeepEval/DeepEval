import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Multiply
from keras.layers import AveragePooling2D, MaxPooling2D, Concatenate, Activation
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 image dimensions
    
    # Initial convolutional layer
    initial_conv = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Block 1
    def block1(input_tensor):
        # Path 1: Global Average Pooling
        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(units=3, activation='relu')(path1)
        path1 = Dense(units=3, activation='sigmoid')(path1)

        # Path 2: Global Max Pooling
        path2 = GlobalMaxPooling2D()(input_tensor)
        path2 = Dense(units=3, activation='relu')(path2)
        path2 = Dense(units=3, activation='sigmoid')(path2)

        # Element-wise addition and apply channel attention weights
        channel_attention = Add()([path1, path2])
        channel_attention = Activation('sigmoid')(channel_attention)
        channel_attention = Multiply()([input_tensor, channel_attention])
        
        return channel_attention
    
    block1_output = block1(initial_conv)
    
    # Block 2
    def block2(input_tensor):
        # Average Pooling
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        
        # Max Pooling
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        
        # Concatenation along the channel dimension
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        
        # 1x1 Convolution and normalization
        conv1x1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(concat)
        
        # Multiply with channel features from Block 1
        spatial_attention = Multiply()([input_tensor, conv1x1])
        
        return spatial_attention
    
    block2_output = block2(block1_output)
    
    # Additional branch with a 1x1 convolutional layer
    conv_branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2_output)
    
    # Adding to the main path
    output = Add()([block1_output, conv_branch])
    output = Activation('relu')(output)
    
    # Final classification layer
    flatten = GlobalAveragePooling2D()(output)  # Flatten the output
    final_output = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=final_output)

    return model