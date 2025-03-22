import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Multiply, Reshape, Add, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    main_path = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)
    
    # Branch path
    global_avg_pooling = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=64, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=32, activation='relu')(dense1)
    channel_weights = Dense(units=3, activation='sigmoid')(dense2)
    reshaped_weights = Reshape(target_shape=(1, 1, 3))(channel_weights)
    weighted_input = Multiply()([input_layer, reshaped_weights])
    
    # Combine paths
    combined = Add()([main_path, weighted_input])
    
    # Additional fully connected layers for classification
    flatten = Flatten()(combined)
    fc1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(fc1)
    
    # Define model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model