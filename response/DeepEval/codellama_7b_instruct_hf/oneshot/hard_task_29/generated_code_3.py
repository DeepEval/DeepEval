from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # First block
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(conv1)
    main_path = Concatenate()([conv1, conv2])
    branch_path = input_layer
    block_output = main_path + branch_path
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    # Second block
    input_layer = Input(shape=(28, 28, 1))
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(maxpool1)
    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(1, 1), padding='same')(maxpool2)
    maxpool_output = Concatenate()([maxpool1, maxpool2, maxpool3])
    flatten_layer = Flatten()(maxpool_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    # Final model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model