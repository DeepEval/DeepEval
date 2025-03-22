import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Four parallel branches
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Block 2: Reduce dimensionality using global average pooling
    global_pool = GlobalAveragePooling2D()(output_tensor)
    
    # Fully connected layer to generate weights
    weight_layer = Dense(units=64, activation='relu')(global_pool)
    
    # Reshape to match the input's shape
    weight_layer = Reshape((64, 1, 1))(weight_layer)
    
    # Element-wise multiplication
    feature_map = Multiply()([output_tensor, weight_layer])
    
    # Final fully connected layer
    output_layer = Dense(units=10, activation='softmax')(feature_map)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model