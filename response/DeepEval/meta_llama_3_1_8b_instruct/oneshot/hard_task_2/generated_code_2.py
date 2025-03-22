import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, Add, Lambda

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def split_input(input_tensor):
        return Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
    
    split_layer = split_input(input_layer)
    
    def group(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        output_tensor = conv3
        return output_tensor
        
    group_output = group(split_layer[0])
    group_output = group(split_layer[1])
    group_output = group(split_layer[2])
    
    main_path = Add()([group_output, group_output, group_output])
    
    combined_features = Add()([main_path, input_layer])
    
    flatten_layer = layers.Flatten()(combined_features)
    output_layer = layers.Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model