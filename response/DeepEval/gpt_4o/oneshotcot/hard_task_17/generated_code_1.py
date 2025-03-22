import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, MaxPooling2D, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: SE-like block
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1_block1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2_block1 = Dense(units=3, activation='sigmoid')(dense1_block1)  # Assuming same channels as input
    weights = Reshape((1, 1, 3))(dense2_block1)
    weighted_feature_output = Multiply()([input_layer, weights])

    # Block 2: Feature extraction block
    conv1_block2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2_block2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1_block2)
    max_pool_block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2_block2)

    # Adding branch connection from Block 1 to Block 2
    added = Add()([max_pool_block2, weighted_feature_output])

    # Fully connected layers after combining
    flatten_layer = Flatten()(added)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model