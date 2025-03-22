import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    initial_conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Main path
    main_branch = initial_conv
    main_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_branch)
    main_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_branch)
    main_branch = BatchNormalization()(main_branch)
    main_branch = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_branch)
    
    # Branch path
    branch_path = initial_conv
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)
    branch_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = BatchNormalization()(branch_path)
    branch_path = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)
    branch_path = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)
    
    # Concatenate outputs
    concat_layer = Concatenate()([main_branch, branch_path])
    
    # Fully connected layers
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model