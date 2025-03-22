import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Add, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    avg_pool_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)

    # Combining input with the output of the main path
    combined = Add()([input_layer, avg_pool_1])

    # Second Block (Squeeze and Excitation)
    global_avg_pool = GlobalAveragePooling2D()(combined)
    dense_se1 = Dense(units=64, activation='relu')(global_avg_pool)
    dense_se2 = Dense(units=combined.shape[-1], activation='sigmoid')(dense_se1)
    reshaped_weights = Reshape((1, 1, combined.shape[-1]))(dense_se2)

    # Scaling the input with the computed weights
    scaled_combined = Multiply()([combined, reshaped_weights])

    # Flatten and final Dense layer for classification
    flatten_layer = Flatten()(scaled_combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model