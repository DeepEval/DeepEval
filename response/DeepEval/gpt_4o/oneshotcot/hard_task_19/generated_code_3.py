import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, Reshape, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)
    main_conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv2)
    main_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(main_conv3)
    
    # Branch Path
    global_avg_pool = GlobalAveragePooling2D()(main_conv3)
    dense1 = Dense(units=16, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32, activation='sigmoid')(dense1)
    channel_weights = Reshape((1, 1, 32))(dense2)
    scaled_features = Multiply()([main_conv3, channel_weights])
    
    # Combine the main path and branch path
    combined = Add()([main_pool, scaled_features])
    
    # Additional Fully Connected Layers for classification
    flatten_layer = Flatten()(combined)
    fc1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(fc1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model