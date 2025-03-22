import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, MaxPooling2D, Flatten, Dense, tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer) 
    
    # Three branches with depthwise separable convolutions
    branch1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    branch2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    branch3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    
    # Concatenate outputs from branches
    x = Concatenate()([branch1, branch2, branch3])
    
    # Second Block
    x = Lambda(lambda x: tf.split(x, 6, axis=-1))(x)
    
    # Branch 1
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
    branch4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch4)
    
    # Branch 2
    branch5 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[1])
    branch5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch5)
    
    # Branch 3
    branch6 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x[2])
    branch6 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch6)
    
    # Concatenate outputs from branches
    x = Concatenate()([branch4, branch5, branch6])
    
    # Flatten and output layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model