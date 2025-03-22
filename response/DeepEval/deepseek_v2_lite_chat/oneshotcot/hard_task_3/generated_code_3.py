import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel dimension
    group1, group2, group3 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Define a sequence of 1x1 convolutions followed by 3x3 convolutions
    def conv_block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=1, activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=3, activation='relu')(conv1)
        return conv2
    
    # Apply the conv_block to each group
    conv_output1 = conv_block(group1)
    conv_output2 = conv_block(group2)
    conv_output3 = conv_block(group3)
    
    # Dropout for feature selection
    drop1 = BatchNormalization()(conv_output1)
    drop2 = BatchNormalization()(conv_output2)
    drop3 = BatchNormalization()(conv_output3)
    
    # Concatenate the outputs from the three groups
    concat = Concatenate()([drop1, drop2, drop3])
    
    # Pathway with 1x1 convolution
    pathway1 = Conv2D(filters=64, kernel_size=1, activation='relu')(concat)
    
    # Branch pathway with 1x1 convolution to match dimensions
    pathway2 = Conv2D(filters=64, kernel_size=1, activation='relu')(concat)
    
    # Addition operation to combine outputs from both pathways
    combined = Add()([pathway1, pathway2])
    
    # Fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(combined)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model