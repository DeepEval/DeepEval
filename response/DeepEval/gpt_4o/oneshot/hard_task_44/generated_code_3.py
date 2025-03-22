import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Dropout
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        # Split the input into three groups along the channel axis
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply convolutions with different kernel sizes
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        
        # Apply dropout
        dropout1 = Dropout(rate=0.3)(conv1x1)
        dropout2 = Dropout(rate=0.3)(conv3x3)
        dropout3 = Dropout(rate=0.3)(conv5x5)
        
        # Concatenate the outputs
        concat_output = Concatenate()([dropout1, dropout2, dropout3])
        return concat_output
    
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 1x1 Convolution followed by 3x3 Convolution
        path2_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2_1)
        
        # Path 3: 1x1 Convolution followed by 5x5 Convolution
        path3_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3_2 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(path3_1)
        
        # Path 4: 3x3 Max Pooling followed by 1x1 Convolution
        path4_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4_2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path4_1)
        
        # Concatenate all paths
        concat_output = Concatenate()([path1, path2_2, path3_2, path4_2])
        return concat_output
    
    block2_output = block2(block1_output)
    
    # Final layers
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model