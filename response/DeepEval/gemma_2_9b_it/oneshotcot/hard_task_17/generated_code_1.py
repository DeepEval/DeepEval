import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Concatenate, Reshape

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Reshape((32, 32, 3))(x)
    weighted_features = x * input_layer

    # Block 2
    block2_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block2_output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block2_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2_output)

    # Fusion
    combined_output = keras.layers.Add()([block2_output, weighted_features])

    # Classification
    output_layer = Dense(units=10, activation='softmax')(combined_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model