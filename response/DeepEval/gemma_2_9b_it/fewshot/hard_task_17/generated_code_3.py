import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3)) 

    # Block 1
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x)
    x = Reshape(target_shape=(32, 32, 3))(x)  
    weighted_features = input_layer * x 

    # Block 2
    y = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    y = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(y)
    y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(y)

    # Branch connection
    z = Add()([weighted_features, y])

    # Final classification layers
    z = Flatten()(z)
    z = Dense(units=128, activation='relu')(z)
    output_layer = Dense(units=10, activation='softmax')(z)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model