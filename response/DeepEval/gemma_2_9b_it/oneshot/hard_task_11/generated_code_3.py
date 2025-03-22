import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main pathway
    x_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer) 

    # Parallel branch
    x_branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_branch2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_branch3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_branch = Concatenate()([x_branch1, x_branch2, x_branch3])

    # Concatenate outputs and further processing
    x = Concatenate()([x_main, x_branch])
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)

    # Direct connection
    x = Add()([input_layer, x])

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model