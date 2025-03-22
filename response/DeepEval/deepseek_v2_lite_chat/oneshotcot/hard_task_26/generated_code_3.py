import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, ZeroPadding2D

def dl_model():     
    # Main Path
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv2)
    
    # Branch Path
    branch_input = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool1)
    branch_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_input)
    branch_upsample = UpSampling2D(size=(2, 2))(branch_conv1)
    branch_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_upsample)
    branch_upsample2 = UpSampling2D(size=(2, 2))(branch_conv2)
    branch_conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_upsample2)
    
    # Concatenate and Process
    concat = Concatenate()([conv2, branch_conv1, branch_conv2, branch_conv3])
    conv4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    flatten = Flatten()(conv4)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model Construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model