import keras
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, concatenate, AveragePooling2D
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels
    
    # First block
    block1_input = Input(shape=input_shape)
    
    # Three parallel paths
    path1 = AveragePooling2D(pool_size=1, strides=1)(block1_input)
    path2 = AveragePooling2D(pool_size=2, strides=2)(block1_input)
    path3 = AveragePooling2D(pool_size=4, strides=4)(block1_input)
    
    # Flatten and concatenate
    pooled_paths = concatenate([Flatten()(path1), Flatten()(path2), Flatten()(path3)])
    
    # Second block
    branch1_input = Input(shape=(4,))  # Fixed shape for block 1 output
    branch2_input = Input(shape=(4,))
    branch3_input = Input(shape=(4,))
    
    # Three branches for feature extraction
    branch1 = Conv2D(64, kernel_size=1, activation='relu')(branch1_input)
    branch2 = Conv2D(64, kernel_size=3, activation='relu')(branch2_input)
    branch3 = AveragePooling2D(pool_size=7)(branch3_input)
    branch4 = Conv2D(64, kernel_size=(1, 7), activation='relu')(branch4_input)
    branch5 = Conv2D(64, kernel_size=3, activation='relu')(branch5_input)
    branch6 = AveragePooling2D(pool_size=(7, 1))(branch6_input)
    
    # Concatenate features from all branches
    extracted_features = concatenate([branch1, branch2, branch3, branch4, branch5, branch6])
    
    # Fully connected layer and reshape
    reshaped_output = Flatten()(extracted_features)
    reshaped_output = Dense(128, activation='relu')(reshaped_output)
    reshaped_output = keras.layers.Reshape((4, 4, 64))(reshaped_output)  # Reshape back to 4x4x64
    
    # Classification head
    classification_output = Dense(10, activation='softmax')(reshaped_output)  # 10 classes for MNIST
    
    # Model combining all components
    model = Model(inputs=[block1_input, branch1_input, branch2_input, branch3_input], outputs=classification_output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create the model
model = dl_model()