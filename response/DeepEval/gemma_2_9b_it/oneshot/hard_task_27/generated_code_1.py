import keras
from keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    depthwise_conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', depth_multiplier=1, activation='relu')(input_layer)
    norm_layer = BatchNormalization()(depthwise_conv)

    fc1 = Dense(units=32, activation='relu')(norm_layer)
    fc2 = Dense(units=32, activation='relu')(fc1)

    # Combine original input with processed features
    combined_features = Add()([input_layer, fc2]) 

    flatten_layer = Flatten()(combined_features)
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model