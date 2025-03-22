import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Main path
    # Split the input into three groups
    split_1 = Lambda(lambda x: x)(inputs)
    split_2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1)[1])(inputs)
    split_3 = Lambda(lambda x: tf.split(x, num_or_size_branches=3, axis=-1)[2])(inputs)
    
    # Process the second group with a 3x3 convolutional layer
    processed_2 = Conv2D(32, (3, 3), activation='relu')(split_2)
    
    # Combine the second group with the third group
    combined_2_3 = Add()([processed_2, split_3])
    
    # Additional 3x3 convolution
    additional_conv = Conv2D(32, (3, 3), activation='relu')(combined_2_3)
    
    # Concatenate the outputs of all three groups
    main_path_output = tf.concat([split_1, processed_2, additional_conv], axis=-1)
    
    # Branch path
    branch_output = Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Fuse the outputs from both paths through addition
    fused_output = Add()([main_path_output, branch_output])
    
    # Flatten the combined output
    flattened_output = Flatten()(fused_output)
    
    # Pass through a fully connected layer
    fc_output = Dense(10, activation='softmax')(flattened_output)
    
    # Create the model
    model = Model(inputs=inputs, outputs=fc_output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
# model = dl_model()
# model.summary()