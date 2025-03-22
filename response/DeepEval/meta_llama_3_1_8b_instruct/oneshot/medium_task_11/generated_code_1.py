import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Multiply, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, MaxPooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the channel attention path
    avg_pool_path = GlobalAveragePooling2D()(conv)
    dense1_avg = Dense(units=128, activation='relu')(avg_pool_path)
    dense2_avg = Dense(units=128, activation='relu')(dense1_avg)
    avg_output = Dense(units=32, activation='sigmoid')(dense2_avg)

    # Define the max pooling path
    max_pool_path = GlobalMaxPooling2D()(conv)
    dense1_max = Dense(units=128, activation='relu')(max_pool_path)
    dense2_max = Dense(units=128, activation='relu')(dense1_max)
    max_output = Dense(units=32, activation='sigmoid')(dense2_max)

    # Add the outputs of the two paths and apply element-wise multiplication to get the channel attention weights
    add_layer = Add()([avg_output, max_output])
    mul_layer = Multiply()([add_layer, conv])
    channel_features = keras.layers.Lambda(lambda x: x * add_layer)(mul_layer)

    # Apply average and max pooling to extract spatial features
    avg_pool = AveragePooling2D(pool_size=(8, 8))(channel_features)
    max_pool = MaxPooling2D(pool_size=(8, 8))(channel_features)

    # Concatenate the spatial features along the channel dimension
    concat_layer = Concatenate()([avg_pool, max_pool])

    # Apply element-wise multiplication to combine the channel and spatial features
    element_wise_mul = Multiply()([channel_features, concat_layer])

    # Flatten the output and pass it through a fully connected layer
    flatten_layer = Flatten()(element_wise_mul)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model