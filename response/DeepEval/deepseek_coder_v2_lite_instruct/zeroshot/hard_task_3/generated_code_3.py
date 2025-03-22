import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, Add, Concatenate, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Define the main pathway
    main_pathway = []
    
    for i, group in enumerate(split):
        # 1x1 Convolution
        x = Conv2D(64, (1, 1), activation='relu')(group)
        # 3x3 Convolution
        x = Conv2D(64, (3, 3), activation='relu')(x)
        main_pathway.append(x)
    
    # Dropout for feature selection
    main_pathway = [Dropout(0.5)(x) for x in main_pathway]
    
    # Concatenate the outputs from the three groups
    main_output = Concatenate(axis=-1)(main_pathway)
    
    # Define the branch pathway
    branch_output = Conv2D(64, (1, 1), activation='relu')(inputs)
    
    # Add the main pathway and the branch pathway
    added_output = Add()([main_output, branch_output])
    
    # Flatten the output
    flattened_output = Flatten()(added_output)
    
    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(flattened_output)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()