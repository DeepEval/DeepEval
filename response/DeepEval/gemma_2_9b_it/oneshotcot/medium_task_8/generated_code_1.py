import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # First Group
    x1 = x[0]

    # Second Group
    x2 = x[1]
    x2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x2)

    # Third Group
    x3 = x[2]
    x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x3)

    # Combine and Concatenate
    x = Concatenate()([x1, x2, x3])

    # Branch Path
    branch = Lambda(lambda x: Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(input_layer)
    
    # Fusion
    x = x + branch

    # Flatten and Classify
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model