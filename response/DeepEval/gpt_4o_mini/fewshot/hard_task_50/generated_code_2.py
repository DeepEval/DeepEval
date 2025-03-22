import keras
import tensorflow as tf
from keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Dropout, Dense, Reshape, Lambda
from keras.layers import SeparableConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First block
    def block_1(input_tensor):
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        
        # Apply dropout to mitigate overfitting
        dropout1 = Dropout(rate=0.5)(flatten1)
        dropout2 = Dropout(rate=0.5)(flatten2)
        dropout3 = Dropout(rate=0.5)(flatten3)

        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    # Second block
    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)
        
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(inputs_groups[3])
        
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    
    # Fully connected layer and reshaping for block 2
    dense = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 12))(dense)  # The output of block1 is flattened and needs to reshape appropriately
    
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model