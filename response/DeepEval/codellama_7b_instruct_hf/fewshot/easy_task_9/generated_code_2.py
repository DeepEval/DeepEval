import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    conv2 = DepthwiseConv2D(32, (3, 3), activation='relu')(conv1)
    conv3 = Conv2D(32, (1, 1), activation='relu')(conv2)
    adding_layer = Add()([input_layer, conv3])
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(10, activation='softmax')(flatten_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model