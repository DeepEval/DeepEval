import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Multiply, Reshape, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    block1_input = input_layer
    global_avg_pool = GlobalAveragePooling2D()(block1_input)
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    weights = Reshape(target_shape=(1, 64))(dense2)
    weighted_block1_output = Multiply()([block1_input, weights])
    
    # Block 2
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_input)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    branch = weighted_block1_output
    
    # Fuse the two paths
    addition = Add()([max_pooling, branch])
    
    # Classifier
    flatten_layer = Flatten()(addition)
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model