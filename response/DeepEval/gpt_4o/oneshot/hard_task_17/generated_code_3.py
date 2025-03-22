import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, MaxPooling2D, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    # Global average pooling to generate weights
    gap = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=32 * 3, activation='relu')(gap)  # Assuming 32 filters, adjust based on filters used
    dense2 = Dense(units=32 * 3, activation='sigmoid')(dense1)  # Scale between 0 and 1
    reshaped_weights = Reshape((1, 1, 32 * 3))(dense2)  # Adjust this based on actual channel size

    # Multiply weights with input feature
    weighted_features = Multiply()([input_layer, reshaped_weights])

    # Block 2
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(weighted_features)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Direct branch from Block 1 to output of Block 2
    # Assuming we adjust the dimensions to match
    adjusted_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(weighted_features)

    # Fuse outputs from main path and branch
    fused_output = Add()([max_pool, adjusted_branch])

    # Final classification layers
    flatten = Flatten()(fused_output)
    dense3 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model