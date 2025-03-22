import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)

    # Branch path
    branch = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Combine paths
    concat = Add()([conv2, branch])

    # Additional processing
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(concat)
    bath_norm = BatchNormalization()(max_pooling)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model