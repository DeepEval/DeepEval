import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_tensor)
        conv1x1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        conv3x3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        conv5x5 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        batch_norm = BatchNormalization()(Concatenate()([conv1x1, conv3x3, conv5x5]))
        return batch_norm
    
    block_output1 = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        path2 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Concatenate()([path3, input_tensor])
        
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Concatenate()([path4, input_tensor])
        
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor
    
    block_output2 = block2(block_output1)
    
    bath_norm = BatchNormalization()(block_output2)
    flatten_layer = Flatten()(bath_norm)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model