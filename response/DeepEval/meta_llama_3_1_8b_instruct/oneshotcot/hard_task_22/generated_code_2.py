import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add
from keras.layers import SeparableConv2D
from keras import backend as K
from tensorflow.keras import layers

def dl_model():
    
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path: multi-scale feature extraction
    def multi_scale_feature_extraction(input_tensor):
        group1 = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        group2 = SeparableConv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        group3 = SeparableConv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        outputs = Concatenate()([group1, group2, group3])
        return outputs
    
    # Split input into three groups along the channel
    split_output = Lambda(lambda x: K.tf.split(x, 3, axis=-1))(input_layer)
    group1, group2, group3 = split_output[0], split_output[1], split_output[2]
    
    # Apply multi-scale feature extraction to each group
    ms_group1 = multi_scale_feature_extraction(group1)
    ms_group2 = multi_scale_feature_extraction(group2)
    ms_group3 = multi_scale_feature_extraction(group3)
    
    # Concatenate the outputs from the three groups
    main_path_output = Concatenate()([ms_group1, ms_group2, ms_group3])
    
    # Branch path: 1x1 convolutional layer
    branch_path = Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse the outputs from the main path and the branch path
    fused_output = Add()([main_path_output, branch_path])
    
    # Batch normalization
    bath_norm = BatchNormalization()(fused_output)
    
    # Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model