import keras
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense, Reshape, Conv2D, MaxPooling2D, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x)
    x = Reshape(target_shape=(32, 32, 3))(x)
    weighted_features = input_layer * x

    # Block 2
    y = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(weighted_features)
    y = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)

    # Fusion
    output = Add()([y, weighted_features])

    # Classification
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model