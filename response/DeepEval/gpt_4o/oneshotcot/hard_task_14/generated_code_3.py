import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path with Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=32 * 3, activation='relu')(global_avg_pool)
    dense2_main = Dense(units=32 * 3, activation='sigmoid')(dense1_main)
    reshape_main = keras.layers.Reshape((1, 1, 32 * 3))(dense2_main)
    scale = Multiply()([input_layer, reshape_main])  # Element-wise multiplication
    
    # Branch path with 3x3 convolution
    branch_conv = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Add the outputs from both paths
    combined_output = Add()([scale, branch_conv])
    
    # Fully connected layers for final classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model