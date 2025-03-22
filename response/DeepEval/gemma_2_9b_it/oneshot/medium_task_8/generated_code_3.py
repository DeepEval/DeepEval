import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    path1 = x[0]
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[2])
    
    combined = Concatenate()([path1, path2, path3])

    # Branch Path
    branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion
    output = keras.layers.add([combined, branch])
    
    # Classification
    flatten_layer = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model