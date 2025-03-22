import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # First block: Splitting input into three groups and applying depthwise separable convolutions
    def split_and_depthwise(input_tensor):
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', depthwise=True)(splits[0])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', depthwise=True)(splits[1])
        path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', depthwise=True)(splits[2])
        return Concatenate()([path1, path2, path3])
    
    block1_output = split_and_depthwise(input_layer)

    # Second block: Multiple branches for feature extraction
    def second_block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        
        path3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path3)
        
        path4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(input_tensor)
        
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor
    
    block2_output = second_block(block1_output)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model