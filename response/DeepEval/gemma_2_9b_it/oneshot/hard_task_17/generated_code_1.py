import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Reshape, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(128, activation='relu')(x)
    x = Dense(3, activation='relu')(x)  # Output channels match input channels
    x = Reshape((32, 32, 3))(x)  # Reshape to match input shape
    weighted_features = input_layer * x  # Element-wise multiplication

    # Block 2
    y = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(weighted_features)
    y = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(y)
    y = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(y)

    # Fusion
    z = Add()([y, weighted_features])

    # Classification
    z = Flatten()(z)
    z = Dense(128, activation='relu')(z)
    output_layer = Dense(units=10, activation='softmax')(z)  

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model