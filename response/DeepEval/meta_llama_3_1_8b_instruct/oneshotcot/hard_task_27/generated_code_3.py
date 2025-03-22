import keras
from keras.layers import Input, DepthwiseConv2D, Activation, BatchNormalization, Add, Dense
from keras.layers import Conv2D, MaxPooling2D, LayerNormalization, Concatenate, Flatten
from keras import regularizers

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same')(input_layer)
    layer_norm = LayerNormalization()(depthwise_conv)
    activation = Activation('relu')(layer_norm)

    # channel-wise feature transformation
    dense1 = Dense(units=128, activation='relu')(activation)
    channel_add = Add()([input_layer, dense1])
    dense2 = Dense(units=64, activation='relu')(channel_add)
    channel_add2 = Add()([input_layer, dense2])

    # combine the original input with the processed features
    channel_add2 = Concatenate()([input_layer, dense2])
    flatten_layer = Flatten()(channel_add2)

    # final two fully connected layers
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    dropout = keras.layers.Dropout(0.2)(dense3)
    output_layer = Dense(units=10, activation='softmax')(dropout)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()