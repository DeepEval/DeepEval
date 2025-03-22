import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, UpSampling2D, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    init_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Main Path
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(init_conv)
    
    branch2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(init_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(3, 3))(branch2)
    
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(init_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(3, 3))(branch3)
    
    outputs = Concatenate()([branch1, branch2, branch3])
    concat_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(outputs)
    
    # Branch Path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(init_conv)
    
    # Add outputs from both paths
    added_outputs = keras.layers.Add()([concat_conv, branch_path])
    
    # Flatten and Dense layers
    bath_norm = BatchNormalization()(added_outputs)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model