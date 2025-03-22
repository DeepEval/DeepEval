import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 convolutions
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    # Branch 2: 1x1 convolutions + 3x3 convolutions
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Branch 3: Max Pooling
    maxpool3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input_layer)

    # Multi-scale feature fusion block
    output_branch1 = conv1
    output_branch2 = conv2
    output_branch3 = maxpool3
    output_tensor = Concatenate()([output_branch1, output_branch2, output_branch3])

    # Average pooling to reduce spatial dimensions
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(output_tensor)
    
    # Batch normalization
    bath_norm = BatchNormalization()(avg_pool)

    # Flatten the output
    flatten_layer = Flatten()(bath_norm)

    # Two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model