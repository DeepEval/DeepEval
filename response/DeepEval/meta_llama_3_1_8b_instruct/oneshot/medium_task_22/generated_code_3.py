import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import AveragePooling2D, GlobalAveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch1)

    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch2)

    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_layer)

    # Create a multi-scale feature fusion block
    output_tensor = Concatenate()([max_pooling1, max_pooling2, branch3])

    # Apply batch normalization and average pooling
    bath_norm = BatchNormalization()(output_tensor)
    avg_pooling = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(bath_norm)

    # Flatten the output
    flatten_layer = Flatten()(avg_pooling)

    # Apply two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model