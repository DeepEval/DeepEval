import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)

    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3])

        return output_tensor
    
    block1_output = block(split[0])
    block2_output = block(split[1])
    block3_output = block(split[2])
    
    fused_output = Concatenate()([block1_output, block2_output, block3_output])
    bath_norm = BatchNormalization()(fused_output)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model