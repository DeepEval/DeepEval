import keras
from keras.layers import Input, AveragePooling2D, Lambda, Flatten, Concatenate, Dropout, Dense, Reshape, DepthwiseConv2D, Conv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        drop1 = Dropout(0.2)(flatten1)
        
        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        drop2 = Dropout(0.2)(flatten2)
        
        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        drop3 = Dropout(0.2)(flatten3)
        
        output_tensor = Concatenate()([drop1, drop2, drop3])
        return output_tensor

    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        
        conv3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv4 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3)
        
        conv5 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv6 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv5)
        
        conv7 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(inputs_groups[3])
        conv8 = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv7)
        
        output_tensor = Concatenate()([conv2, conv4, conv6, conv8])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model