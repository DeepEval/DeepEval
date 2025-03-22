import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Concatenate, Dense, Reshape

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = GlobalAveragePooling2D()(input_layer)
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=32, activation='relu')(x)
    x = Dense(units=3, activation='linear')(x) # Match channels of input layer
    x = Reshape((32, 32, 3))(x) # Reshape to match input layer's shape
    x = keras.layers.Multiply()([input_layer, x]) # Element-wise multiplication

    # Branch Path
    branch_x = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate outputs
    x = Concatenate()([x, branch_x])

    # Classification Layers
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model