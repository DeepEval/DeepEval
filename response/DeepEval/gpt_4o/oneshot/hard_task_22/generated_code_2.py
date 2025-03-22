import keras
from keras.layers import Input, Lambda, SeparableConv2D, Conv2D, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    def multi_scale_feature_extraction(input_tensor):
        split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layers[0])
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layers[1])
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layers[2])
        
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor
    
    main_path_output = multi_scale_feature_extraction(input_layer)
    
    # Branch Path
    branch_path_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse Paths
    fused_output = Add()([main_path_output, branch_path_output])
    
    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model