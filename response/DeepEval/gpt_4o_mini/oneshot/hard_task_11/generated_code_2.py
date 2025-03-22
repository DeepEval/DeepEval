import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, BatchNormalization, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Pathway for 1x1 convolution
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Parallel path with 1x1, 1x3, and 3x1 convolutions
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Concatenate outputs of parallel paths
    concatenated_path = Concatenate()([path1, path2, path3])
    
    # Convolution to match input channel dimensions
    conv_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated_path)

    # Add input layer to the main output (additive connection)
    added_output = Add()([main_path, conv_output])

    # Batch normalization
    batch_norm = BatchNormalization()(added_output)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model