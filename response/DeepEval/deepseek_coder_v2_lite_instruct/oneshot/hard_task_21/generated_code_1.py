import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = []
    for kernel_size in [(1, 1), (3, 3), (5, 5)]:
        conv = Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(input_layer)
        main_path.append(conv)
    
    main_output = Concatenate(axis=-1)(main_path)
    
    # Branch path
    branch_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs of main and branch paths
    added_output = Add()([main_output, branch_conv])
    
    # Batch normalization and flatten
    batch_norm = BatchNormalization()(added_output)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model