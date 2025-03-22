import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        # Split the input into three groups along the channel axis
        split_1 = Lambda(lambda x: tf.split(x, 3, axis=3))(input_tensor)
        split_2 = Lambda(lambda x: tf.split(x, 3, axis=3))(input_tensor)
        split_3 = Lambda(lambda x: tf.split(x, 3, axis=3))(input_tensor)
        
        # Feature extraction through convolutional layers
        conv_1x1 = Conv2D(32, (1, 1), activation='relu')(split_1[0])
        conv_3x3 = Conv2D(32, (3, 3), activation='relu')(split_1[1])
        conv_5x5 = Conv2D(32, (5, 5), activation='relu')(split_1[2])
        
        # Apply dropout to reduce overfitting
        dropout = Dropout(0.25)(Concatenate(axis=3)([conv_1x1, conv_3x3, conv_5x5]))
        
        return dropout
    
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        # Four branches
        path1 = Conv2D(32, (1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(32, (1, 1), activation='relu')(input_tensor)
        path3 = Conv2D(32, (1, 1), activation='relu')(input_tensor)
        path4 = MaxPooling2D((3, 3), strides=1, padding='same')(input_tensor)
        
        # Convolutional layers with 1x1 kernel
        conv_1x1_1 = Conv2D(32, (1, 1), activation='relu')(path1)
        conv_1x1_2 = Conv2D(32, (1, 1), activation='relu')(path2)
        conv_1x1_3 = Conv2D(32, (1, 1), activation='relu')(path3)
        
        # Convolutional layers with varying kernel sizes
        conv_3x3 = Conv2D(32, (3, 3), activation='relu')(path4)
        conv_5x5 = Conv2D(32, (5, 5), activation='relu')(path4)
        
        # Concatenate the outputs from all branches
        concat = Concatenate(axis=-1)([conv_1x1_1, conv_1x1_2, conv_1x1_3, conv_3x3, conv_5x5])
        
        return concat
    
    block2_output = block2(block1_output)
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layer
    dense_layer = Dense(10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=dense_layer)
    
    return model