import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Convolutional Layer
    conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=None)(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)
    
    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(relu)
    
    # Fully Connected Layers for Channel Attention
    dense1 = Dense(units=64//2, activation='relu')(global_avg_pool)  # Reduce dimensions
    dense2 = Dense(units=64, activation='sigmoid')(dense1)  # Output same dimension as initial features
    channel_weights = Reshape((1, 1, 64))(dense2)  # Reshape to match initial feature size
    
    # Weighted Feature Maps
    weighted_features = Multiply()([relu, channel_weights])
    
    # Concatenate with Input Layer
    concatenated = Concatenate()([input_layer, weighted_features])
    
    # Dimensionality Reduction and Downsampling
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1x1)
    
    # Output Layer
    flatten = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model