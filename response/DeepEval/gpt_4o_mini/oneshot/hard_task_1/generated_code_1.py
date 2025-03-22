import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, Concatenate
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels (RGB)
    
    # Initial convolutional layer to adjust output channels
    conv_initial = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Block 1: Parallel paths for channel attention
    # Path 1: Global Average Pooling
    gap = GlobalAveragePooling2D()(conv_initial)
    dense1_path1 = Dense(64, activation='relu')(gap)
    dense2_path1 = Dense(64, activation='relu')(dense1_path1)

    # Path 2: Global Max Pooling
    gmp = GlobalMaxPooling2D()(conv_initial)
    dense1_path2 = Dense(64, activation='relu')(gmp)
    dense2_path2 = Dense(64, activation='relu')(dense1_path2)

    # Add outputs of both paths
    channel_attention = Add()([dense2_path1, dense2_path2])
    channel_attention = Activation('sigmoid')(channel_attention)
    
    # Reshape to match the input feature map shape for multiplication
    channel_attention = keras.layers.Reshape((1, 1, 64))(channel_attention)
    
    # Apply channel attention to the original features
    attention_output = Multiply()([conv_initial, channel_attention])
    
    # Block 2: Spatial feature extraction
    avg_pool = GlobalAveragePooling2D()(attention_output)
    max_pool = GlobalMaxPooling2D()(attention_output)
    
    # Concatenate the outputs along the channel dimension
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])
    
    # 1x1 Convolution to normalize the features
    spatial_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='sigmoid')(attention_output)
    
    # Multiply spatial features with channel attention features
    normalized_spatial_features = Multiply()([spatial_features, spatial_conv])
    
    # Additional branch to ensure output channels align with input channels
    conv_branch = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(normalized_spatial_features)
    
    # Add branch output to the main path
    final_output = Add()([attention_output, conv_branch])
    final_output = Activation('relu')(final_output)

    # Flatten and fully connected layer for classification
    flatten_layer = keras.layers.Flatten()(final_output)
    classification_output = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=classification_output)

    return model