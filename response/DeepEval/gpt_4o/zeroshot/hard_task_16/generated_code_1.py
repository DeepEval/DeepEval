import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Concatenate, GlobalMaxPooling2D, Dense, Reshape, Add
from tensorflow.keras.models import Model

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Split input into three groups
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)

    # Use Lambda to split the input
    splits = Lambda(split_input)(input_layer)
    
    # Process each split through a series of convolutions
    conv_outputs = []
    for split in splits:
        x = Conv2D(32, (1, 1), activation='relu')(split)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (1, 1), activation='relu')(x)
        conv_outputs.append(x)
    
    # Concatenate the outputs from the three convolutions
    concat_output = Concatenate()(conv_outputs)
    
    # Transition Convolution: Adjust channels to match input
    trans_conv = Conv2D(3, (1, 1), padding='same', activation='relu')(concat_output)

    # Block 2: Global Max Pooling
    pooled = GlobalMaxPooling2D()(trans_conv)
    
    # Channel matching weights with fully connected layers
    fc1 = Dense(10, activation='relu')(pooled)  # Arbitrary intermediate size
    fc2 = Dense(3, activation='sigmoid')(fc1)   # Match the channels of the input (3 channels)
    
    # Reshape weights to match adjusted output
    weights = Reshape((1, 1, 3))(fc2)
    
    # Scale adjusted output by weights
    scaled_output = tf.keras.layers.multiply([trans_conv, weights])
    
    # Branch from input directly
    branch_output = input_layer
    
    # Add the scaled output and the branch output
    added_output = Add()([scaled_output, branch_output])
    
    # Final fully connected layer for classification
    flatten = tf.keras.layers.Flatten()(added_output)
    final_output = Dense(10, activation='softmax')(flatten)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=final_output)
    
    return model

# Create the model
model = dl_model()

# Summary of the model
model.summary()