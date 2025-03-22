import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 convolutions
    conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)

    # Branch 2: 5x5 convolutions
    conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2_1)

    # Combine branches with addition
    combined = Add()([conv1_2, conv2_2])

    # Global average pooling
    pool = GlobalAveragePooling2D()(combined)

    # Attention mechanism
    attention_dense1 = Dense(units=128, activation='relu')(pool)
    attention_dense2 = Dense(units=10, activation='softmax')(attention_dense1)  

    # Weighted sum of branches
    weighted_output = keras.layers.multiply([conv2_2, attention_dense2]) + keras.layers.multiply([conv1_2, 1 - attention_dense2])

    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(weighted_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model