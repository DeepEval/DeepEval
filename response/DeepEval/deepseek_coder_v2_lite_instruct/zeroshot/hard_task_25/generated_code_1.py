import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main path
    x = Conv2D(64, (1, 1), activation='relu')(inputs)
    
    # First branch (local features)
    branch1 = Conv2D(32, (3, 3), activation='relu')(x)
    
    # Second branch (downsample and process)
    branch2 = AveragePooling2D((2, 2))(x)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = UpSampling2D((2, 2))(branch2)
    
    # Third branch (downsample and process)
    branch3 = AveragePooling2D((2, 2))(x)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D((2, 2))(branch3)
    
    # Concatenate the outputs of all branches
    merged = Concatenate()([branch1, branch2, branch3])
    
    # Apply a 1x1 convolutional layer to form the main path output
    main_path_output = Conv2D(64, (1, 1), activation='relu')(merged)
    
    # Branch path
    branch_path_output = Conv2D(64, (1, 1), activation='relu')(inputs)
    
    # Fuse the main path and branch path outputs through addition
    fused_output = tf.add(main_path_output, branch_path_output)
    
    # Flatten the output
    flattened = Flatten()(fused_output)
    
    # Fully connected layer for 10-class classification
    outputs = Dense(10, activation='softmax')(flattened)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()