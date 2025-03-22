import keras
from keras.layers import Input, Conv2D, MaxPooling2D, DepthwiseConv2D, Dropout, Flatten, Dense, Concatenate

def block(input_tensor):
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    dropout = Dropout(0.5)(conv)
    return dropout

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    branch1_output = block(input_tensor)
    branch2_output = block(branch1_output)
    branch3_output = block(branch2_output)

    concat = Concatenate()([branch1_output, branch2_output, branch3_output])

    flatten = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model