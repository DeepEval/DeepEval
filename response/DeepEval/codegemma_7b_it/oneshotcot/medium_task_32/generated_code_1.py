import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def depthwise_separable_conv(input_tensor, kernel_size, padding):
        conv1 = Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding=padding, depthwise_initializer='he_normal', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal', activation='relu')(conv1)
        return conv2
    
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    conv_1x1 = depthwise_separable_conv(split_input[0], kernel_size=1, padding='valid')
    conv_3x3 = depthwise_separable_conv(split_input[1], kernel_size=3, padding='same')
    conv_5x5 = depthwise_separable_conv(split_input[2], kernel_size=5, padding='same')
    
    concat_outputs = Concatenate()([conv_1x1, conv_3x3, conv_5x5])
    
    flatten_layer = Flatten()(concat_outputs)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model