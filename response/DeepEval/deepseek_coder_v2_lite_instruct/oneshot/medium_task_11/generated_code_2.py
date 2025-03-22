import keras
from keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # First parallel path: Global Average Pooling followed by two fully connected layers
    gap_path = AveragePooling2D(pool_size=(8, 8))(conv1)
    gap_flatten = Flatten()(gap_path)
    gap_dense1 = Dense(units=64, activation='relu')(gap_flatten)
    gap_dense2 = Dense(units=32, activation='relu')(gap_dense1)
    
    # Second parallel path: Global Max Pooling followed by two fully connected layers
    gmp_path = MaxPooling2D(pool_size=(8, 8))(conv1)
    gmp_flatten = Flatten()(gmp_path)
    gmp_dense1 = Dense(units=64, activation='relu')(gmp_flatten)
    gmp_dense2 = Dense(units=32, activation='relu')(gmp_dense1)
    
    # Concatenate outputs from both paths and apply activation to generate channel attention weights
    concat_attention = Concatenate()([gap_dense2, gmp_dense2])
    activation_layer = keras.activations.sigmoid(concat_attention)
    
    # Apply channel attention weights to the original features
    channel_attention = Multiply()([conv1, activation_layer])
    
    # Separate average and max pooling operations to extract spatial features
    avg_pool = AveragePooling2D(pool_size=(8, 8))(channel_attention)
    max_pool = MaxPooling2D(pool_size=(8, 8))(channel_attention)
    
    # Concatenate spatial features along the channel dimension
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])
    
    # Flatten the fused feature map
    flatten_layer = Flatten()(spatial_features)
    
    # Fully connected layer to produce the final output
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model