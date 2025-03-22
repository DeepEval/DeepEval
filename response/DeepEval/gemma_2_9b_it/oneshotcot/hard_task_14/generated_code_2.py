import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(units=32, activation='relu')(x) 
    x = Dense(units=3, activation='relu')(x)  
    x = keras.layers.Reshape((32, 32, 3))(x)  
    x = keras.layers.multiply([input_layer, x])

    # Branch Path
    branch_x = Conv2D(filters=3, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_layer)

    # Concatenate Paths
    combined_features = Concatenate()([x, branch_x])

    # Fully Connected Layers
    x = Flatten()(combined_features)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model