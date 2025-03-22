import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Depthwise Separable Convolutions
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensor[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensor[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_tensor[2])

    # Concatenate the outputs of the main path
    main_path_output = Concatenate()([path1, path2, path3])
    
    # Branch Path
    branch_path_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add main path output and branch path output
    combined_output = Add()([main_path_output, branch_path_output])

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model