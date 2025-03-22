import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Branch Path
    branch_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch_max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_conv1)

    # Combine Outputs
    combined_output = Add()([max_pooling2, branch_max_pooling])

    # Batch Normalization and Flattening
    bath_norm = BatchNormalization()(combined_output)
    flatten_layer = Flatten()(bath_norm)

    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()