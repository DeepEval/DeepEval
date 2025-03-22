import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Concatenate, Dropout, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)
    
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])

    x = Concatenate()([conv1, conv2, conv3])
    x = Dropout(0.25)(x)

    # Block 2
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
    x = Concatenate()([branch1, branch2, branch3])

    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model