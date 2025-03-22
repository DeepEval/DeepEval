import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense, concatenate
from keras.models import Model

def dl_model():

    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv)
        output_tensor = Add()([input_tensor, conv2])
        return output_tensor

    input_layer = Input(shape=(28, 28, 1))
    
    branch1 = block(input_tensor=input_layer)
    branch2 = block(input_tensor=input_layer)
    branch3 = block(input_tensor=input_layer)

    outputs = concatenate([branch1, branch2, branch3])
    flatten_layer = Flatten()(outputs)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=dense2)

    return model

dl_model = dl_model()
dl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])