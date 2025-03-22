import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Splitting the input into three groups along the channel dimension
    groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    def conv_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3
    
    # Apply the conv_block to each group
    group_outputs = [conv_block(group) for group in groups]
    
    # Combine the outputs of the three groups using an addition operation
    main_path = Add()(group_outputs)
    
    # Fuse the main path with the original input layer
    fused = Add()([main_path, input_layer])
    
    # Flatten and connect to a fully connected layer for classification
    flatten_layer = Flatten()(fused)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model