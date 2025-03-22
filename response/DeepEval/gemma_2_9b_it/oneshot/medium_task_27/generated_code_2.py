import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Add, Flatten, Dense, Activation

def dl_model(): 
    
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    conv1_1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1_1)

    # Branch 2: 5x5 convolutions
    conv2_1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
    conv2_2 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(conv2_1)

    # Combine branches
    merged = Add()([conv1_2, conv2_2])

    # Global average pooling
    pool = GlobalAveragePooling2D()(merged)

    # Attention layers
    attention1 = Dense(units=10, activation='softmax')(pool)
    attention2 = Dense(units=10, activation='softmax')(pool)

    # Weighted sum
    weighted_output = keras.layers.multiply([attention1, conv1_2]) + keras.layers.multiply([attention2, conv2_2])

    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(weighted_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model