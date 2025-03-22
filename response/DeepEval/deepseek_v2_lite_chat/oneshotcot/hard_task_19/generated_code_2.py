import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # Convolutional layers
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv2)
        
        # Max pooling layer
        maxpool = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        return maxpool
    
    # Branch path
    def branch_path(input_tensor):
        # Global average pooling
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        
        # Fully connected layers for channel weights
        fc1 = Dense(units=512, activation='relu')(avg_pool)
        fc2 = Dense(units=256, activation='relu')(fc1)
        
        # Reshape and multiply with input
        channel_weights = keras.layers.Lambda(lambda x: keras.layers.Reshape(target_shape=(1, 1, x.shape[1]))(x))(keras.layers.Lambda(lambda x: tf.reduce_prod(x))(fc2))
        output_tensor = keras.layers.Multiply()([input_tensor, channel_weights])
        
        return output_tensor
    
    # Concatenate outputs from main path and branch path
    main_output = main_path(input_tensor=input_layer)
    branch_output = branch_path(input_tensor=input_layer)
    merged_output = Concatenate()([main_output, branch_output])
    
    # Additional fully connected layers
    dense1 = Dense(units=256, activation='relu')(merged_output)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model