import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model(): 
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 convolutions
    conv1_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_3x3)

    # Branch 2: 5x5 convolutions
    conv1_5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv1_5x5)

    # Combine branches
    merged_features = Add()([conv2_3x3, conv2_5x5])

    # Global average pooling
    pool_output = GlobalAveragePooling2D()(merged_features)

    # Fully connected layers with softmax for attention weights
    att_1 = Dense(units=128, activation='relu')(pool_output)
    att_2 = Dense(units=10, activation='softmax')(att_1)

    # Weighted sum of branch outputs
    weighted_output = Multiply()([conv2_3x3, att_2[:, :, None, None]]) + Multiply()([conv2_5x5, att_2[:, :, None, None]])

    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(weighted_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model