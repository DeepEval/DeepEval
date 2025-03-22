import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Conv2DTranspose
from keras.models import Model

# Define the input shape
input_shape = (28, 28, 1)

# Define the first special block
def block1(input_tensor):
    # Define the parallel paths
    path1 = Conv2D(32, (1, 1), activation='relu')(input_tensor)
    path2 = Conv2D(64, (2, 2), activation='relu')(input_tensor)
    path3 = Conv2D(128, (4, 4), activation='relu')(input_tensor)
    
    # Flatten and concatenate the outputs
    path1 = Flatten()(path1)
    path2 = Flatten()(path2)
    path3 = Flatten()(path3)
    output = Concatenate()([path1, path2, path3])
    
    # Apply dropout
    output = Dropout(0.2)(output)
    
    return output

# Define the second special block
def block2(input_tensor):
    # Define the branches
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_tensor)
    branch2 = Conv2D(64, (1, 1), activation='relu')(input_tensor)
    branch3 = Conv2D(128, (3, 3), activation='relu')(input_tensor)
    branch4 = Conv2D(256, (5, 5), activation='relu')(input_tensor)
    
    # Concatenate the outputs
    branch1 = Flatten()(branch1)
    branch2 = Flatten()(branch2)
    branch3 = Flatten()(branch3)
    branch4 = Flatten()(branch4)
    output = Concatenate()([branch1, branch2, branch3, branch4])
    
    return output

# Define the model
input_layer = Input(shape=input_shape)
output = block1(input_layer)
output = block2(output)
output = Flatten()(output)
output = Dense(128, activation='relu')(output)
output = Dense(10, activation='softmax')(output)
model = Model(inputs=input_layer, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])