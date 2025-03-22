import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    
    # Main path
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)
    
    # Adding the input to the main path
    add_output = Add()([input_layer, main_path])
    
    # Second block
    block_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(add_output)
    block_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_output)
    global_avg_pool = GlobalAveragePooling2D()(block_output)
    
    # Refining the global average pooling output
    channel_weights = Dense(units=32, activation='relu')(global_avg_pool)
    channel_weights = Reshape((1, 1, 32))(channel_weights)
    
    # Multiplying the channel weights with the input
    multiplied_output = Multiply()([channel_weights, block_output])
    
    # Flattening the output
    flatten_layer = Flatten()(multiplied_output)
    
    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model