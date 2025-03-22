import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling)
    
    # Branch path
    global_avg_pooling = GlobalAveragePooling2D()(conv3)
    dense1 = Dense(units=128, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=64, activation='relu')(dense2)
    channel_weights = Dense(units=128, activation='linear')(dense2)
    
    # Reshape channel weights
    channel_weights = Reshape((128, 1))(channel_weights)
    
    # Multiply channel weights with input
    multiply = Multiply()([conv3, channel_weights])
    
    # Concatenate main path output with branch path output
    concat_output = Add()([conv3, multiply])
    
    # Flatten the output
    flatten_layer = Flatten()(concat_output)
    
    # Classification
    dense3 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model