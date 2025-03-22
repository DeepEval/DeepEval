import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv)
    main_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_branch)
    
    main_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_branch)
    main_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_branch)
    main_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_branch)
    
    main_branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_branch)
    concat = Concatenate()([main_branch_output, main_conv])
    
    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    branch_up = UpSampling2D(size=(2, 2))(branch_conv)
    branch_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_up)
    branch_down = UpSampling2D(size=(2, 2))(branch_conv)
    
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_down)
    add = keras.layers.add([concat, branch_output])
    
    # Fully connected layers
    fully_connected = Dense(units=128, activation='relu')(add)
    fully_connected = Dense(units=64, activation='relu')(fully_connected)
    output_layer = Dense(units=10, activation='softmax')(fully_connected)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()