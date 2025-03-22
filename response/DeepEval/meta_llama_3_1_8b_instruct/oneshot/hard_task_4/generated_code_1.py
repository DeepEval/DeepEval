import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, GlobalAveragePooling2D, Reshape, Multiply, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=3*32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dw_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv2)

    def channel_attention(input_tensor):

        pool = GlobalAveragePooling2D()(input_tensor)
        fc1 = Dense(units=128, activation='relu')(pool)
        fc2 = Dense(units=32, activation='sigmoid')(fc1)
        reshape = Reshape((1, 1, 32))(fc2)
        multiply = Multiply()([input_tensor, reshape])
        return multiply
    
    attention_output = channel_attention(conv2)
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(attention_output)
    concat = Concatenate()([conv1, conv3])

    batch_norm = BatchNormalization()(concat)
    flatten_layer = Flatten()(batch_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model