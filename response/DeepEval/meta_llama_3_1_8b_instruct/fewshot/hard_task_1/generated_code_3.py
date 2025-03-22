import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Multiply, Dense, Lambda, Reshape

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def path1(input_tensor):
        gavgpool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=64, activation='relu')(gavgpool)
        dense2 = Dense(units=10, activation='softmax')(dense1)
        return dense2

    def path2(input_tensor):
        gmaxpool = GlobalMaxPooling2D()(input_tensor)
        dense1 = Dense(units=64, activation='relu')(gmaxpool)
        dense2 = Dense(units=10, activation='softmax')(dense1)
        return dense2

    path1_output = path1(input_tensor=conv)
    path2_output = path2(input_tensor=conv)

    addition = Add()([path1_output, path2_output])
    activation = Lambda(lambda x: keras.activations.relu(x))(addition)

    # Channel attention weights
    channel_weights = Multiply()([activation, activation])
    
    # Block 2
    block2_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block2_conv)
    max_pool = GlobalMaxPooling2D()(block2_conv)

    concat = Concatenate()([avg_pool, max_pool])
    block2_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat)

    # Normalize spatial features
    sigmoid = keras.layers.Activation('sigmoid')(block2_conv)

    # Element-wise multiplication
    element_wise_mult = Multiply()([channel_weights, sigmoid])

    # Additional branch
    additional_conv = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(element_wise_mult)

    # Final classification
    final_output = Add()([element_wise_mult, additional_conv])
    classification = Dense(units=10, activation='softmax')(final_output)

    model = keras.Model(inputs=input_layer, outputs=classification)

    return model