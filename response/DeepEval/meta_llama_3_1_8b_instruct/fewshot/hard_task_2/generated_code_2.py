import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Lambda, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def group_operation(input_tensor):
        group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        group1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1)
        group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        
        group2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
        group2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group2)
        
        group3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        group3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group3)
        group3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group3)
        
        output_tensor = Add()([group1, group2, group3])
        return output_tensor
    
    def fuse_operation(input_tensor):
        fused_tensor = Add()([input_tensor, input_layer])
        return fused_tensor
    
    group1_output = group_operation(input_layer)
    group2_output = group_operation(group1_output)
    group3_output = group_operation(group2_output)
    
    fused_output = fuse_operation(group3_output)
    
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model