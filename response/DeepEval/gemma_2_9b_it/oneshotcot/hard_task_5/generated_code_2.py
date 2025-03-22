import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute
from tensorflow import tf

def dl_model(): 
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    x = [Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(xi) for xi in x]
    x = Concatenate(axis=3)(x)
    
    # Block 2
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(x)
    x = [tf.reshape(xi, (-1, 32, 32, 3, 1)) for xi in x]
    x = [Permute((1, 2, 4, 3))(xi) for xi in x]
    x = [tf.reshape(xi, (-1, 32, 32, 9)) for xi in x]
    x = Concatenate(axis=3)(x)
    
    # Block 3
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', depth_wise=True)(x)
    shortcut = input_layer
    x = x + shortcut
    
    # Final Layer
    x = Flatten()(x)
    x = Dense(units=10, activation='softmax')(x) 

    model = keras.Model(inputs=input_layer, outputs=x)
    
    return model