import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Conv2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    def main_path(input_tensor):
        
        block1 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        block2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2)
        
        return max_pooling
    
    def branch_path(input_tensor):
        
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        return conv1
    
    block1_output = main_path(input_layer)
    block2_output = branch_path(block1_output)
    summed_output = Add()([block1_output, block2_output])
    bath_norm = BatchNormalization()(summed_output)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model