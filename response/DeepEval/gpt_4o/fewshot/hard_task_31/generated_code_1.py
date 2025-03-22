import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    def block_1(input_tensor):
        # Main Path
        conv_main = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        dropout_main = Dropout(rate=0.3)(conv_main)
        conv_restore = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(dropout_main)
        
        # Branch Path
        branch_path = input_tensor
        
        # Addition of Main and Branch Paths
        output_tensor = Add()([conv_restore, branch_path])
        return output_tensor
    
    # Second Block
    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        dropout1 = Dropout(rate=0.3)(conv1)
        
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        dropout2 = Dropout(rate=0.3)(conv2)
        
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        dropout3 = Dropout(rate=0.3)(conv3)
        
        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    # Apply blocks
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)

    # Final classification layers
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model