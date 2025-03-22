import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # First block
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Split branches
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])

    batch_norm = BatchNormalization()(branch1)
    batch_norm = BatchNormalization()(branch2)
    batch_norm = BatchNormalization()(branch3)

    # Concatenate branches
    x = Concatenate()([batch_norm, batch_norm, batch_norm])
    
    # Second block
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch5 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(x)
    branch6 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch7 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    branch8 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Concatenate branches
    x = Concatenate()([branch4, branch5, branch6, branch7, branch8])

    # Classification layers
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model