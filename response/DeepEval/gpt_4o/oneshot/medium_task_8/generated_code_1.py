import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    def main_path(input_tensor):
        # Split the input into three groups along the last dimension
        group1, group2, group3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_tensor)
        
        # First group remains unchanged
        unchanged = group1
        
        # Second group undergoes a 3x3 convolution
        conv_group2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(group2)
        
        # Combine the output of second group with the third group and pass through another 3x3 convolution
        combined = tf.concat([conv_group2, group3], axis=3)
        conv_combined = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(combined)
        
        # Concatenate the outputs of all three groups
        main_output = tf.concat([unchanged, conv_group2, conv_combined], axis=3)
        
        return main_output
    
    # Branch Path
    def branch_path(input_tensor):
        branch_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        return branch_output
    
    # Get outputs from both paths
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    
    # Fuse outputs from both paths through addition
    fused_output = Add()([main_output, branch_output])
    
    # Flatten and fully connected layer for final classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model