import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense, Reshape, Permute

def dl_model(): 
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial Convolutional Layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Block 1
    x = Lambda(lambda x: tf.split(x, 2, axis=3))(conv1)
    
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch1)

    branch2 = x[1]

    output_block1 = Concatenate()([branch1, branch2])
    
    # Block 2
    x = Lambda(lambda x: tf.reshape(x, (-1, 14, 14, 64 * 2)))(output_block1)
    x = Permute((1, 2, 4, 3))(x)
    x = Reshape((14 * 14, 64 * 2))(x)
    
    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model