import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Main path
    # Split the input into three groups along the channel
    main_path_groups = tf.split(inputs, num_or_size_splits=3, axis=-1)
    
    # Multi-scale feature extraction
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(main_path_groups[0])
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path_groups[1])
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(main_path_groups[2])
    
    # Concatenate the outputs from the three groups
    main_path_output = tf.concat([conv1x1, conv3x3, conv5x5], axis=-1)
    
    # Branch path
    branch_path_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs)
    
    # Fuse the outputs from both paths through addition
    fused_output = Add()([main_path_output, branch_path_output])
    
    # Flatten the output
    flattened_output = Flatten()(fused_output)
    
    # Pass through two fully connected layers for classification
    fc1 = Dense(units=128, activation='relu')(flattened_output)
    fc2 = Dense(units=10, activation='softmax')(fc1)
    
    # Define the model
    model = Model(inputs=inputs, outputs=fc2)
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()