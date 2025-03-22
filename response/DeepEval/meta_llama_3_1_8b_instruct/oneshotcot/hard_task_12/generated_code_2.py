import keras
from keras.layers import Input, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    reduction_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reduction_conv)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reduction_conv)
    main_output = Concatenate()([path1, path2])

    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add main and branch outputs
    add_output = Add()([main_output, branch_conv])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(add_output)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model