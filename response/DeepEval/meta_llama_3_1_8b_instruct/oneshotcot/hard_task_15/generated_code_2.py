import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool = GlobalAveragePooling2D()(conv)
    dense1 = Dense(units=128, activation='relu')(pool)
    weights = Dense(units=3, activation='linear')(dense1)
    weights_reshape = Reshape(target_shape=(1, 1, 3))(weights)
    weighted_conv = Multiply()([conv, weights_reshape])
    
    # Branch Path
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine Main and Branch Path
    combine = Add()([weighted_conv, branch_conv])
    
    # Flatten the combined result
    flatten_layer = Flatten()(combine)
    
    # Add two fully connected layers
    dense2 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model