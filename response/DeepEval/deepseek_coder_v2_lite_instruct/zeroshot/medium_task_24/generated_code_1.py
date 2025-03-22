import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)
    
    # Branch 2
    branch2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(32, (1, 7), activation='relu')(branch2)
    branch2 = Conv2D(32, (7, 1), activation='relu')(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    
    # Branch 3
    branch3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch3)
    
    # Concatenate outputs from all branches
    combined = concatenate([branch1, branch2, branch3])
    
    # Flatten the concatenated output
    combined_flat = Flatten()(combined)
    
    # Fully connected layers
    fc1 = Dense(128, activation='relu')(combined_flat)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(64, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)
    output_layer = Dense(10, activation='softmax')(fc2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Note: The CIFAR-10 dataset is included in Keras, so you don't need to download it separately.
# Ensure you preprocess the data correctly before training the model.