import keras
from keras.layers import Input, Lambda, SeparableConv2D, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    def first_block(input_tensor):
        # Splitting input into three groups along the channel dimension
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
        
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    first_block_output = first_block(input_layer)
    
    # Second block
    def second_block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)

        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)

        path3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)

        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    second_block_output = second_block(first_block_output)
    
    # Global average pooling and dense layers
    global_avg_pool = GlobalAveragePooling2D()(second_block_output)
    output_layer = Dense(units=10, activation='softmax')(global_avg_pool)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model