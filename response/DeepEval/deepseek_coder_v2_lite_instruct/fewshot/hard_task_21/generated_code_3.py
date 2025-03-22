import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    main_path_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(main_path_inputs[0])
    conv1_3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path_inputs[1])
    conv1_5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(main_path_inputs[2])
    
    main_path_output = Concatenate(axis=-1)([conv1_1, conv1_3, conv1_5])
    
    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add both paths
    added_output = Add()([main_path_output, branch_path])
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model