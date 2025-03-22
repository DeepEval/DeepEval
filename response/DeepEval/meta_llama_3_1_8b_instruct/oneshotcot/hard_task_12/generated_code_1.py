import keras
from keras.layers import Input, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 64))
    
    # Main Path
    dimension_reduction = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dimension_reduction)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dimension_reduction)
    main_path_output = Concatenate()([path1, path2])

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine Main and Branch Paths
    combined_output = Add()([main_path_output, branch_path])

    # Batch Normalization
    batch_norm = BatchNormalization()(combined_output)

    # Flatten
    flatten_layer = Flatten()(batch_norm)

    # Dense Layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=64, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model