import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Squeeze and Excitation Block
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=input_layer.shape[-1], activation='relu')(global_avg_pool)
    dense2 = Dense(units=input_layer.shape[-1], activation='sigmoid')(dense1)
    reshaped_weights = Reshape((1, 1, input_layer.shape[-1]))(dense2)
    weighted_features = Multiply()([input_layer, reshaped_weights])
    
    # Block 2: Convolutional Block
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(weighted_features)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    
    # Fusion through addition
    fused_output = Add()([weighted_features, max_pool])
    
    # Flatten and Dense Layers
    flatten_layer = Flatten()(fused_output)
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model