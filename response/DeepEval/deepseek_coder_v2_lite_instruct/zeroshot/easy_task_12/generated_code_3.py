import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Define the main path
    input_main = Input(shape=(28, 28, 1))
    
    # First block in the main path
    x_main = Conv2D(32, (3, 3), activation='relu')(input_main)
    x_main = SeparableConv2D(64, (3, 3), activation='relu')(x_main)
    x_main = MaxPooling2D((2, 2))(x_main)
    
    # Second block in the main path
    x_main = Conv2D(64, (3, 3), activation='relu')(x_main)
    x_main = SeparableConv2D(128, (3, 3), activation='relu')(x_main)
    x_main = MaxPooling2D((2, 2))(x_main)
    
    # Define the branch path
    input_branch = Input(shape=(28, 28, 1))
    x_branch = Conv2D(32, (1, 1), activation='relu')(input_branch)
    
    # Add the main path and branch path
    x = Add()([x_main, x_branch])
    
    # Flatten the output
    x = Flatten()(x)
    
    # Fully connected layer
    output = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=[input_main, input_branch], outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage
model = dl_model()
model.summary()