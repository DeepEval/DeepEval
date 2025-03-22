import keras
from keras.layers import Input, Conv2D, Lambda, Dropout, Concatenate, Add, Dense, Flatten
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    processed_splits = []
    for split_layer in split_layers:
        # Apply 1x1 convolution followed by a 3x3 convolution
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split_layer)
        conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1x1)
        # Apply dropout
        dropout = Dropout(rate=0.5)(conv3x3)
        processed_splits.append(dropout)
    
    # Concatenate the processed splits
    main_pathway = Concatenate()(processed_splits)
    
    # Branch pathway to match the main pathway dimension
    branch_pathway = Conv2D(filters=192, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    
    # Combine main pathway and branch pathway using addition
    combined = Add()([main_pathway, branch_pathway])
    
    # Flatten the combined output
    flatten_layer = Flatten()(combined)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model