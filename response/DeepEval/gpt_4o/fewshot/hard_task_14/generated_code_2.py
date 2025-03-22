import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Conv2D, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    global_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=64, activation='relu')(global_pool)
    dense2 = Dense(units=3, activation='relu')(dense1)  # Assumes channel size is 3
    reshaped_weights = Dense(units=(32 * 32 * 3), activation='relu')(dense2)  # Adjust shape
    reshaped_weights = keras.layers.Reshape(target_shape=(32, 32, 3))(reshaped_weights)
    weighted_feature_map = Multiply()([input_layer, reshaped_weights])
    
    # Branch path
    conv = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine paths
    combined_output = Add()([weighted_feature_map, conv])
    
    # Fully connected layers
    flatten = Flatten()(combined_output)
    fc1 = Dense(units=128, activation='relu')(flatten)
    fc2 = Dense(units=64, activation='relu')(fc1)
    output_layer = Dense(units=10, activation='softmax')(fc2)  # CIFAR-10 has 10 classes
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model