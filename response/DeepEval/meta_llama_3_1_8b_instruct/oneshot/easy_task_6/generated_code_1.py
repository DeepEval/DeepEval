import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv_main1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_main2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_main1)

    # Branch path
    branch = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the two paths through addition
    combined = Add()([conv_main2, branch])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model