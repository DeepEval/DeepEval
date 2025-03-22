import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv_path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    max_pooling_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(max_pooling_path)
    
    def block(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv1)
        maxpool = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])

        return output_tensor
    
    block_output1 = block(input_tensor=conv_path1)
    block_output2 = block(input_tensor=conv_path2)
    block_output3 = block(input_tensor=conv_path3)
    block_output4 = block(input_tensor=max_pooling_conv)
    
    bath_norm = BatchNormalization()(block_output1)
    bath_norm2 = BatchNormalization()(block_output2)
    bath_norm3 = BatchNormalization()(block_output3)
    bath_norm4 = BatchNormalization()(block_output4)
    
    flatten_layer = Flatten()(bath_norm)
    flatten_layer2 = Flatten()(bath_norm2)
    flatten_layer3 = Flatten()(bath_norm3)
    flatten_layer4 = Flatten()(bath_norm4)
    
    concat = Concatenate()([flatten_layer, flatten_layer2, flatten_layer3, flatten_layer4])
    dense1 = Dense(units=128, activation='relu')(concat)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model