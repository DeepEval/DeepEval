import keras
from keras.layers import Input, Conv2D, Lambda, Add, Dense, Flatten
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Main path: Depthwise separable convolutions with varying kernel sizes
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal')(split_tensors[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', depthwise_initializer='he_normal')(split_tensors[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', depthwise_initializer='he_normal')(split_tensors[2])
    
    # Concatenate the outputs from the main path
    main_path_output = tf.concat([path1, path2, path3], axis=-1)
    
    # Branch path: 1x1 convolution to align channels
    branch_path_output = Conv2D(filters=main_path_output.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs of main and branch paths
    combined_output = Add()([main_path_output, branch_path_output])
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model