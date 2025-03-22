import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def main_path(input_tensor):
        # Global average pooling to extract global information
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        
        # Two fully connected layers to generate weights
        weights = Dense(32, activation='relu')(avg_pool)
        weights = Dense(3, activation='relu')(weights)
        
        # Reshape weights to match input layer's shape
        weights = Reshape(target_shape=(1, 1, 3))(weights)
        
        # Multiply element-wise with input feature map
        multiplied = Multiply()([input_tensor, weights])
        
        return multiplied
    
    def branch_path(input_tensor):
        return input_tensor
    
    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)
    
    # Add outputs from both paths
    added = Add()([main_path_output, branch_path_output])
    
    # Flatten output
    flatten = Flatten()(added)
    
    # Two fully connected layers to produce final probability distribution
    output_layer = Dense(10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model