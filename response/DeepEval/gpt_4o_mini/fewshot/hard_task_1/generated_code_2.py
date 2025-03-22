import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, Concatenate, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial convolutional layer to adjust channels
    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Block 1: Two parallel processing paths
    # Path 1: Global Average Pooling and two fully connected layers
    avg_pool = GlobalAveragePooling2D()(conv_initial)
    dense1_avg = Dense(units=64, activation='relu')(avg_pool)
    dense2_avg = Dense(units=32, activation='relu')(dense1_avg)

    # Path 2: Global Max Pooling and two fully connected layers
    max_pool = GlobalMaxPooling2D()(conv_initial)
    dense1_max = Dense(units=64, activation='relu')(max_pool)
    dense2_max = Dense(units=32, activation='relu')(dense1_max)

    # Combine outputs from both paths
    combined = Add()([dense2_avg, dense2_max])
    channel_attention_weights = Activation('sigmoid')(combined)

    # Apply channel attention weights to original features
    attention_output = Multiply()([conv_initial, channel_attention_weights])

    # Block 2: Extract spatial features through pooling
    avg_pool_block2 = GlobalAveragePooling2D()(attention_output)
    max_pool_block2 = GlobalMaxPooling2D()(attention_output)
    spatial_features = Concatenate()([avg_pool_block2, max_pool_block2])

    # 1x1 Convolution followed by sigmoid activation
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='sigmoid')(attention_output)

    # Normalize features and multiply with the channel features from Block 1
    normalized_features = Multiply()([spatial_features, conv1x1])

    # Final 1x1 convolution to ensure output channels align
    output_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(normalized_features)

    # Adding back to the initial path
    final_output = Add()([attention_output, output_conv])
    final_activation = Activation('relu')(final_output)

    # Flatten and classification
    flatten_layer = Flatten()(final_activation)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model