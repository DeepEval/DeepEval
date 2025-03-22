import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path - split input into three groups and apply different convolutions
    def split_and_convolve(input_tensor):
        # Split the input into three groups along the channel dimension
        split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply convolutions with different kernel sizes
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensor[0])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensor[1])
        path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_tensor[2])
        
        # Concatenate the outputs
        output_tensor = Concatenate()([path1, path2, path3])
        
        return output_tensor
    
    main_path_output = split_and_convolve(input_layer)
    
    # Branch path - align number of output channels
    branch_path_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the main and branch paths
    fused_features = Add()([main_path_output, branch_path_output])
    
    # Classification part
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model