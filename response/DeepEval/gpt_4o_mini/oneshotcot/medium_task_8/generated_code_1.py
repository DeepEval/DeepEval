import keras
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main Path
    def main_path(input_tensor):
        # Split the input into three groups along the last dimension
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Group 1 remains unchanged
        group1 = groups[0]
        
        # Group 2 undergoes feature extraction
        group2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(groups[1])
        
        # Combine Group 2 with Group 3
        combined_group = Add()([group2, groups[2]])
        
        # Additional 3x3 convolution
        main_output = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(combined_group)
        
        # Concatenate all groups
        output_tensor = Concatenate()([group1, group2, main_output])
        
        return output_tensor

    # Create the main path output
    main_output = main_path(input_layer)
    
    # Branch Path
    branch_output = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse the main path and branch path outputs
    fused_output = Add()([main_output, branch_output])
    
    # Flatten the fused output
    flatten_layer = Flatten()(fused_output)
    
    # Fully connected layer for classification
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)  # 10 classes for CIFAR-10
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model