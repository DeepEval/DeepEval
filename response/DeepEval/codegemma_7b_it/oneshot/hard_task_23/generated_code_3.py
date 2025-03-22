import keras
from keras.layers import Input, Conv2D, AveragePooling2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Convolution
    conv_init = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Local Feature Extraction Branch
    branch_local = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_init)
    branch_local = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_local)
    
    # Downsampling Branch
    branch_down = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_init)
    branch_down = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_down)
    
    # Upsampling Branch
    branch_up = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_init)
    branch_up = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_up)
    branch_up = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch_up)
    
    # Concatenation and Refinement
    concat = Concatenate()([branch_local, branch_down, branch_up])
    concat = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    
    # Fully Connected Layer
    flatten = Flatten()(concat)
    dense = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()