import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Multiply, Reshape, Add, Flatten

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Squeeze-and-Excitation (SE) Block
    se_gap = GlobalAveragePooling2D()(input_layer)
    se_dense1 = Dense(units=32, activation='relu')(se_gap)
    se_dense2 = Dense(units=96, activation='sigmoid')(se_dense1)  # 96 is the same as the channel size of the input
    
    # Reshape and multiply with the input layer
    se_reshape = Reshape((1, 1, 96))(se_dense2)
    se_weighted_output = Multiply()([input_layer, se_reshape])
    
    # Block 2: Feature extraction block with branch connection
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(se_weighted_output)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Addition of the direct branch from Block 1
    branch_addition = Add()([max_pool, se_weighted_output])
    
    # Final classification layers
    flatten_layer = Flatten()(branch_addition)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create the Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model