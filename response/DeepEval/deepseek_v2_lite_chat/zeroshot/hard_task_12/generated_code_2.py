import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, concatenate, Flatten, Dense
from tensorflow.keras.models import Model


def dl_model(input_shape=(32, 32, 64)):
    # Define the main path
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (1, 1), activation='relu')(inputs)  # 1x1 convolutional layer for dimensionality reduction
    x1 = Conv2D(64, (1, 1), activation='relu')(inputs)
    x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    
    # Concatenate the outputs of the main path
    concat = concatenate([x, x1, x2])
    
    # Process the branch path
    branch_input = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    
    # Combine the outputs from the main and branch paths
    combined = Add()([concat, branch_input])
    
    # Flatten the output for fully connected layers
    flat = Flatten()(combined)
    
    # Fully connected layers for classification
    fc1 = Dense(512, activation='relu')(flat)
    output = Dense(10, activation='softmax')(fc1)  # Assuming 10 classes for image classification
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model