import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, MaxPooling2D, Add, Flatten

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    global_avg_pooling = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=32, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=32, activation='relu')(dense1)
    reshaped_weights = Reshape((1, 1, 32))(dense2)
    weighted_feature_output = Multiply()([input_layer, reshaped_weights])
    
    # Block 2
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(weighted_feature_output)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Branch from Block 1 directly to output of Block 2
    branch_connection = Reshape((16, 16, 32))(dense2)
    
    # Fuse outputs from main path and branch
    added = Add()([max_pooling, branch_connection])
    
    # Classification with two fully connected layers
    flatten_layer = Flatten()(added)
    fc1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(fc1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model