import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Reshape((32, 32, 3))(x)  
    weighted_features = input_layer * x

    # Block 2
    block2_out = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block2_out = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_out)
    block2_out = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2_out)

    # Fusion
    x = Add()([weighted_features, block2_out])

    # Classification layers
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model