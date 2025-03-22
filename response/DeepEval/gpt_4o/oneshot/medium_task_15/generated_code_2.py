import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Convolutional Layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)
    
    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(relu)
    
    # Two Fully Connected Layers
    dense1 = Dense(units=32 // 2, activation='relu')(global_avg_pool)  # Reduce dimensions
    dense2 = Dense(units=32, activation='relu')(dense1)  # Match initial features' channels
    
    # Reshape and Multiply
    reshape = Reshape((1, 1, 32))(dense2)
    weighted_features = Multiply()([relu, reshape])
    
    # Concatenate with Input Layer
    concatenated = Concatenate()([input_layer, weighted_features])
    
    # Dimensionality Reduction and Downsampling
    conv_reduce = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_reduce)
    
    # Final Fully Connected Layer
    final_dense = Dense(units=10, activation='softmax')(avg_pool)

    # Model
    model = keras.Model(inputs=input_layer, outputs=final_dense)
    
    return model