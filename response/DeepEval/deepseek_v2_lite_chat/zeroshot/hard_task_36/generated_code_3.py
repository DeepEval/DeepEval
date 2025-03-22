from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Dense, Flatten
from keras.layers import Layer

def dl_model():
    # Define the paths for the two pathways
    input_pathway = Input(shape=(28, 28, 1))
    branch_pathway = Input(shape=(8, 8, 32))  # Output feature maps from the main pathway

    # Main pathway
    x = Conv2D(32, (3, 3), activation='relu')(input_pathway)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout(0.5)(x)
    
    # Branch pathway
    branch_x = Conv2D(32, (1, 1), activation='relu')(branch_pathway)
    
    # Concatenate the outputs from both pathways
    combined_output = concatenate([x, branch_x])
    
    # Global average pooling, flattening, and fully connected layer
    combined_output = Flatten()(combined_output)
    output = Dense(10, activation='softmax')(combined_output)
    
    # Create the model
    model = Model(inputs=[input_pathway, branch_pathway], outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Instantiate and return the model
model = dl_model()