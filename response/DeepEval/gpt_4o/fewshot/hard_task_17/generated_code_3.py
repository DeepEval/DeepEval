import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, MaxPooling2D, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    gap = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=input_layer.shape[-1], activation='relu')(gap)
    dense2 = Dense(units=input_layer.shape[-1], activation='sigmoid')(dense1)
    weights = Reshape((1, 1, input_layer.shape[-1]))(dense2)
    weighted_feature = Multiply()([input_layer, weights])
    
    # Block 2
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(weighted_feature)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Branch connection from Block 1
    branch_connection = Reshape((32, 32, 3))(weighted_feature)  # Adjust the shape to match the main path if necessary
    added = Add()([max_pool, branch_connection])
    
    # Final fully connected layers
    flatten = Flatten()(added)
    dense3 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model