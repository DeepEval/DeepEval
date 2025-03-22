import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    # Main path
    main_conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_dropout1 = Dropout(rate=0.3)(main_conv1)
    main_conv2 = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(main_dropout1)
    
    # Branch path
    branch_path = input_layer
    
    # Adding outputs of both paths
    first_block_output = Add()([main_conv2, branch_path])
    
    # Second Block
    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        group1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        dropout1 = Dropout(rate=0.3)(group1)
        
        group2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs_groups[1])
        dropout2 = Dropout(rate=0.3)(group2)
        
        group3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(inputs_groups[2])
        dropout3 = Dropout(rate=0.3)(group3)
        
        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor
    
    second_block_output = block_2(input_tensor=first_block_output)
    
    # Output layers
    flatten = Flatten()(second_block_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model