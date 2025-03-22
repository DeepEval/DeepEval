import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Lambda, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Block 1: Split into three groups and apply convolutions
    split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    conv_blocks = []
    for split in split_1:
        conv_1x1 = Conv2D(32, (1, 1), activation='relu')(split)
        conv_3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_1x1)
        conv_5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv_1x1)
        conv_blocks.append(conv_3x3)
        conv_blocks.append(conv_5x5)
    
    # Concatenate the outputs of the three groups
    concat_1 = Concatenate(axis=-1)(conv_blocks)
    
    # Dropout layer to reduce overfitting
    dropout_1 = Dropout(0.5)(concat_1)
    
    # Block 2: Four branches
    branch_1 = Conv2D(32, (1, 1), activation='relu')(inputs)
    
    branch_2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(branch_2)
    
    branch_3 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch_3 = Conv2D(32, (5, 5), activation='relu', padding='same')(branch_3)
    
    branch_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch_4 = Conv2D(32, (1, 1), activation='relu')(branch_4)
    
    # Concatenate the outputs from all branches
    concat_2 = Concatenate(axis=-1)([branch_1, branch_2, branch_3, branch_4])
    
    # Flatten the output and add a fully connected layer
    flatten = Flatten()(concat_2)
    dense = Dense(128, activation='relu')(flatten)
    
    # Define the output layer
    outputs = Dense(10, activation='softmax')(dense)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()
model.summary()