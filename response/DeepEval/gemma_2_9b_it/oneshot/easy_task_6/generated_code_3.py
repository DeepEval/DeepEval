import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():  
    
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # Branch path
    branch_x = input_layer

    # Combine paths
    x = Add()([x, branch_x])

    # Flatten and fully connected layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model