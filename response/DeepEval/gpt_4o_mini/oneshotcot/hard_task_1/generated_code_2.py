import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have a shape of 32x32x3

    # Initial convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Block 1
    path1_avg_pool = GlobalAveragePooling2D()(conv1)
    dense1_path1 = Dense(units=32, activation='relu')(path1_avg_pool)
    dense2_path1 = Dense(units=32, activation='relu')(dense1_path1)

    path2_max_pool = GlobalMaxPooling2D()(conv1)
    dense1_path2 = Dense(units=32, activation='relu')(path2_max_pool)
    dense2_path2 = Dense(units=32, activation='relu')(dense1_path2)

    # Extracting channel attention weights
    channel_weights = Add()([dense2_path1, dense2_path2])
    channel_weights = Activation('sigmoid')(channel_weights)
    channel_weights = Reshape((1, 1, 32))(channel_weights)  # Reshape to match the channels

    # Applying channel attention to original features
    channel_attention = Multiply()([conv1, channel_weights])

    # Block 2
    avg_pool_block2 = GlobalAveragePooling2D()(channel_attention)
    max_pool_block2 = GlobalMaxPooling2D()(channel_attention)
    
    # Concatenate average and max pooled features
    concat_block2 = Concatenate()([avg_pool_block2, max_pool_block2])

    # 1x1 convolution to normalize features
    conv_block2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='sigmoid')(channel_attention)

    # Combining features with element-wise multiplication
    normalized_features = Multiply()([concat_block2, conv_block2])

    # Ensuring output channels align with input channels
    final_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(normalized_features)
    
    # Adding the final output from block 1 and the final conv layer
    added_output = Add()([final_conv, channel_attention])
    activated_output = Activation('relu')(added_output)

    # Final classification through a fully connected layer
    flatten_layer = GlobalAveragePooling2D()(activated_output)
    final_dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=final_dense)

    return model