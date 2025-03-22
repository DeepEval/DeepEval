import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    def main_path_block(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_tensor)
        return Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5])
    
    main_path_outputs = [main_path_block(input_tensor) for input_tensor in main_path_inputs]
    main_path_output = Add()(main_path_outputs)
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fusion
    fused_output = Add()([main_path_output, branch_path])
    
    # Flatten and fully connected layers
    flattened_output = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model