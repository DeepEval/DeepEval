import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras import layers

def dl_model():  
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)

    # Block 1
    x = Lambda(lambda x: tf.split(x, 2, axis=-1))(conv_layer)
    
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x[0])
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch1)
    
    branch2 = x[1] 

    block1_output = Concatenate()([branch1, branch2])

    # Block 2
    shape = layers.Lambda(lambda x: tf.shape(x))(block1_output)
    reshaped = layers.Reshape(target_shape=(shape[1], shape[2], 4, int(shape[3] / 4)))(block1_output)
    permuted = layers.Permute((2, 3, 1, 4))(reshaped)
    block2_output = layers.Reshape(target_shape=(shape[1], shape[2], shape[3]))(permuted)

    flatten = Flatten()(block2_output)
    dense_layer = Dense(units=10, activation='softmax')(flatten) 

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model