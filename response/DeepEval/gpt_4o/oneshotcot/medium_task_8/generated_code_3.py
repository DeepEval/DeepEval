import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    # Split input into three groups along the last dimension
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    split_layer = Lambda(split_input)(input_layer)
    group1 = split_layer[0]
    group2 = split_layer[1]
    group3 = split_layer[2]
    
    # Feature extraction on second group
    conv_group2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(group2)
    
    # Combine second group output with third group and apply another convolution
    combined_group23 = Concatenate(axis=-1)([conv_group2, group3])
    conv_combined = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(combined_group23)
    
    # Concatenate all three groups
    main_path_output = Concatenate(axis=-1)([group1, conv_group2, conv_combined])
    
    # Branch Path
    branch_path_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse main and branch paths
    fused_output = Add()([main_path_output, branch_path_output])
    
    # Final classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model