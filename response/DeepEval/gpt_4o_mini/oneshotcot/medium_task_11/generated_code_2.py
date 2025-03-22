import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Concatenate, Flatten
from keras.models import Model
from keras.activations import sigmoid

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Add convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Two parallel paths
    # Path 1: Global Average Pooling followed by two fully connected layers
    global_avg_pool = GlobalAveragePooling2D()(conv)
    dense1_avg = Dense(units=128, activation='relu')(global_avg_pool)
    dense2_avg = Dense(units=64, activation='relu')(dense1_avg)

    # Path 2: Global Max Pooling followed by two fully connected layers
    global_max_pool = GlobalMaxPooling2D()(conv)
    dense1_max = Dense(units=128, activation='relu')(global_max_pool)
    dense2_max = Dense(units=64, activation='relu')(dense1_max)

    # Step 4: Combine paths
    combined = Concatenate()([dense2_avg, dense2_max])

    # Step 5: Channel attention weights
    channel_attention = Dense(units=conv.shape[-1], activation='sigmoid')(combined)
    channel_attention = Reshape((1, 1, conv.shape[-1]))(channel_attention)  # Reshape to match input shape

    # Step 6: Apply channel attention weights
    attention_output = Multiply()([conv, channel_attention])

    # Step 7: Extract spatial features via average and max pooling
    spatial_avg_pool = GlobalAveragePooling2D()(attention_output)
    spatial_max_pool = GlobalMaxPooling2D()(attention_output)

    # Step 8: Concatenate spatial features
    spatial_features = Concatenate()([spatial_avg_pool, spatial_max_pool])

    # Step 9: Flatten and feed into a fully connected layer
    flatten_layer = Flatten()(spatial_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model