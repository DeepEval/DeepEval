import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    # Define the main path
    main_input = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation='relu')(main_input)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Define the branch path
    branch_input = Input(shape=(32, 32, 3))
    y = Conv2D(16, (5, 5), activation='relu')(branch_input)
    
    # Concatenate the outputs of both paths
    combined = concatenate([x, y])
    
    # Flatten the combined features
    z = Flatten()(combined)
    
    # Add fully connected layers
    z = Dense(128, activation='relu')(z)
    output = Dense(10, activation='softmax')(z)
    
    # Define the model
    model = Model(inputs=[main_input, branch_input], outputs=output)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()