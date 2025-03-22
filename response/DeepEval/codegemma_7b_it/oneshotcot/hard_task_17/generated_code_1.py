import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    # Block 1
    conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    avg_pooling = GlobalAveragePooling2D()(max_pooling)
    dense1 = Dense(units=max_pooling.shape[1] * max_pooling.shape[2] * max_pooling.shape[3])(avg_pooling)
    dense2 = Dense(units=max_pooling.shape[1] * max_pooling.shape[2] * max_pooling.shape[3])(dense1)
    dense2_reshape = Reshape((max_pooling.shape[1], max_pooling.shape[2], max_pooling.shape[3]))(dense2)
    weighted_output = Multiply()([max_pooling, dense2_reshape])
    # Block 2
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(weighted_output)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2_2)
    # Branch from Block 1
    branch_output = Conv2D(filters=max_pooling2.shape[3], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(weighted_output)
    # Fuse outputs
    added_output = Add()([branch_output, max_pooling2])
    # Output layers
    flatten_layer = Flatten()(added_output)
    dense3 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model