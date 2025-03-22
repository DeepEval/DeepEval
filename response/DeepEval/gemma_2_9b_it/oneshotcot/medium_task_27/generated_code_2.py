import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Flatten

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 convolutions
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)

    # Branch 2: 5x5 convolutions
    conv2_1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2_1)

    # Combine branches using addition
    merged_features = Add()([conv1_2, conv2_2])

    # Global average pooling
    global_pool = GlobalAveragePooling2D()(merged_features)

    # Attention weights
    attention1 = Dense(units=1, activation='softmax')(global_pool)
    attention2 = Dense(units=1, activation='softmax')(global_pool)

    # Weighted output
    weighted_output = attention1 * conv1_2 + attention2 * conv2_2

    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(Flatten()(weighted_output))

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model