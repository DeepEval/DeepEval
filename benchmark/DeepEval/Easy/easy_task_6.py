from keras.models import Model
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=input_layer.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    added = Add()([conv2, input_layer])

    flat = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flat)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
