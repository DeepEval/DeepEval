import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Branch path
    branch_layer = input_layer

    # Combine paths
    combined_layer = Add()([conv2, branch_layer])

    flatten_layer = Flatten()(combined_layer)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model