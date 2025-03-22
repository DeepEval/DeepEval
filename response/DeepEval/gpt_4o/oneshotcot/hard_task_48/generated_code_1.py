import keras
from keras.layers import Input, Lambda, SeparableConv2D, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        # Split input into three groups
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Path 1: SeparableConv2D with 1x1 kernel size
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        path1 = BatchNormalization()(path1)
        
        # Path 2: SeparableConv2D with 3x3 kernel size
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        path2 = BatchNormalization()(path2)
        
        # Path 3: SeparableConv2D with 5x5 kernel size
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        path3 = BatchNormalization()(path3)
        
        # Concatenate paths
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        # Path 1: 1x1 Conv
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 3x3 AveragePooling followed by 1x1 Conv
        path2 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path2)
        
        # Path 3: 1x1 Conv, then split and process with 1x3 and 3x1 Convs
        path3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        subpath3_1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path3)
        subpath3_2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path3)
        path3 = Concatenate()([subpath3_1, subpath3_2])
        
        # Path 4: 1x1 Conv, then 3x3 Conv, split and process with 1x3 and 3x1 Convs
        path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path4)
        subpath4_1 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', activation='relu')(path4)
        subpath4_2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', activation='relu')(path4)
        path4 = Concatenate()([subpath4_1, subpath4_2])
        
        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block2_output = block2(block1_output)

    # Final layers for classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model