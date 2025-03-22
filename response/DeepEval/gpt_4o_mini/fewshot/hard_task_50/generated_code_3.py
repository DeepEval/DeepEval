import keras
import tensorflow as tf
from keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Dropout, Dense, Reshape, Lambda
from keras.layers import SeparableConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are of shape (32, 32, 3)

    # Block 1
    def block_1(input_tensor):
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        
        # Applying Dropout to mitigate overfitting
        flatten1 = Dropout(0.5)(flatten1)
        flatten2 = Dropout(0.5)(flatten2)
        flatten3 = Dropout(0.5)(flatten3)

        # Concatenating the flattened outputs
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    # Block 2
    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)
        
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(inputs_groups[3])
        
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(8, 8, 4))(dense)  # Reshape to a 4D tensor for the second block
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model