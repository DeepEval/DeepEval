import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block(input_tensor):

        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(units=input_tensor.shape[3], activation='relu')(path1)
        path2 = Dense(units=input_tensor.shape[3], activation='sigmoid')(path1)
        path3 = Reshape(target_shape=(input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))(path2)
        path4 = Multiply()([path3, input_tensor])

        return path4
        
    branch_output = block(input_tensor=max_pooling)
    main_output = max_pooling
    combined_output = Add()([main_output, branch_output])
    bath_norm = BatchNormalization()(combined_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model