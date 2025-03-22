import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Flatten, Concatenate, Dense, Reshape, Multiply

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Apply an initial convolutional layer
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define two parallel paths
    def path_1(input_tensor):
        global_pool = Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))(input_tensor)
        flatten = Flatten()(global_pool)
        output = Dense(units=128, activation='relu')(flatten)
        output = Dense(units=10, activation='softmax')(output)
        return output

    def path_2(input_tensor):
        max_pool = Lambda(lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True))(input_tensor)
        flatten = Flatten()(max_pool)
        output = Dense(units=128, activation='relu')(flatten)
        output = Dense(units=10, activation='softmax')(output)
        return output

    # Calculate channel attention weights
    channel_path_1 = path_1(conv)
    channel_path_2 = path_2(conv)
    adding_layer = Add()([channel_path_1, channel_path_2])
    output_tensor = Activation('sigmoid')(adding_layer)

    # Apply element-wise multiplication to calculate the final channel attention weights
    channel_weights = Multiply()([output_tensor, conv])

    # Extract spatial features
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_weights)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_weights)

    # Concatenate spatial features along the channel dimension
    fused_features = Concatenate()([avg_pool, max_pool])

    # Combine spatial features with channel features
    combined_features = Multiply()([fused_features, channel_weights])

    # Flatten the combined features
    flatten_layer = Flatten()(combined_features)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model