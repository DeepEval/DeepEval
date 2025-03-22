import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, MaxPooling2D, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    gap = GlobalAveragePooling2D()(input_layer)
    dense1_block1 = Dense(units=32, activation='relu')(gap)
    dense2_block1 = Dense(units=3, activation='relu')(dense1_block1)
    weights = Reshape((1, 1, 3))(dense2_block1)
    weighted_features = Multiply()([input_layer, weights])
    
    # Block 2
    conv1_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(weighted_features)
    conv2_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_block2)
    max_pool_block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2_block2)
    
    # Branch from Block 1 to Block 2
    branch = GlobalAveragePooling2D()(weighted_features)
    branch = Reshape((1, 1, 3))(branch)
    branch = Reshape((16, 16, 3))(branch)  # Reshape to match the output size of Block 2
    
    # Fusion through addition
    fused_output = Add()([max_pool_block2, branch])
    
    # Fully Connected Layers for classification
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model