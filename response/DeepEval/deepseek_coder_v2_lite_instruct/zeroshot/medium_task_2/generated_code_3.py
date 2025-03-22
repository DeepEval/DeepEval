import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    # Define the main path
    main_input = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation='relu')(main_input)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Define the branch path
    branch_input = Input(shape=(32, 32, 3))
    y = Conv2D(32, (5, 5), activation='relu')(branch_input)
    
    # Concatenate the outputs from both paths
    combined = concatenate([x, y])
    
    # Flatten the combined output
    combined_flat = Flatten()(combined)
    
    # Add fully connected layers
    fc1 = Dense(128, activation='relu')(combined_flat)
    output = Dense(10, activation='softmax')(fc1)
    
    # Create the model
    model = Model(inputs=[main_input, branch_input], outputs=output)
    
    return model