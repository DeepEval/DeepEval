import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # First Block
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    branch1 = BatchNormalization()(branch1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    branch2 = BatchNormalization()(branch2)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    branch3 = BatchNormalization()(branch3)
    
    x = Concatenate()([branch1, branch2, branch3])

    # Second Block
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch4 = BatchNormalization()(branch4)
    branch4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch4)

    branch5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch5 = BatchNormalization()(branch5)
    branch5 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch5)
    branch5 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch5)
    branch5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch5)

    branch6 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Concatenate()([branch4, branch5, branch6])

    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model