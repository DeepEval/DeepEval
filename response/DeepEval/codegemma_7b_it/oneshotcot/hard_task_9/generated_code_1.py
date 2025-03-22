import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1x1)
    conv3_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_3x3)
    concat_main = Concatenate()([conv1_1x1, conv2_3x3, conv3_3x3])
    conv4_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat_main)
    
    # Branch
    branch_conv1_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_conv2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_conv1_1x1)
    
    # Fusion
    concat_fused = add([conv4_1x1, branch_conv2_3x3])
    
    # Classification
    bath_norm = BatchNormalization()(concat_fused)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model