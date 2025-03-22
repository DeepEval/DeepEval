import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        # Split channels into three groups
        group1, group2, group3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply convolutions with different kernel sizes
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(group1)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(group2)
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(group3)
        
        # Apply dropout to reduce overfitting
        dropout1 = Dropout(rate=0.3)(conv1)
        dropout2 = Dropout(rate=0.3)(conv2)
        dropout3 = Dropout(rate=0.3)(conv3)
        
        # Concatenate results
        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        # Branch 1: 1x1 Convolution
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        
        # Branch 2: 1x1 Convolution followed by 3x3 Convolution
        branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch2)
        
        # Branch 3: 1x1 Convolution followed by 5x5 Convolution
        branch3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(branch3)
        
        # Branch 4: 3x3 Max Pooling followed by 1x1 Convolution
        branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(branch4)
        
        # Concatenate all branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor
    
    block2_output = block2(block1_output)
    
    # Final layers
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model