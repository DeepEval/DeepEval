import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, Concatenate, Flatten
from keras.models import Model

def dl_model():
    # Step 1: Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Step 3: Two parallel paths
    # Path 1: Global Average Pooling followed by two Dense layers
    avg_pool = GlobalAveragePooling2D()(conv1)
    dense_avg1 = Dense(units=64, activation='relu')(avg_pool)
    dense_avg2 = Dense(units=32, activation='relu')(dense_avg1)
    
    # Path 2: Global Max Pooling followed by two Dense layers
    max_pool = GlobalMaxPooling2D()(conv1)
    dense_max1 = Dense(units=64, activation='relu')(max_pool)
    dense_max2 = Dense(units=32, activation='relu')(dense_max1)

    # Step 4: Add the outputs of both paths
    channel_features = Add()([dense_avg2, dense_max2])
    channel_weights = Activation('sigmoid')(channel_features)

    # Step 5: Apply channel attention weights to the original features
    channel_attention = Multiply()([conv1, channel_weights])

    # Step 6: Average and Max pooling to extract spatial features
    spatial_avg = GlobalAveragePooling2D()(channel_attention)
    spatial_max = GlobalMaxPooling2D()(channel_attention)

    # Step 7: Concatenate the spatial features along the channel dimension
    spatial_features = Concatenate()([spatial_avg, spatial_max])

    # Step 8: Flatten and fully connected layer
    flatten_layer = Flatten()(spatial_features)
    dense_output = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = Model(inputs=input_layer, outputs=dense_output)

    return model