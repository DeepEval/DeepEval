import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    # Split input into three groups along the channel
    groups = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    
    # Extract features using convolutional layers with varying kernel sizes
    group1 = Conv2D(filters=16, kernel_size=(1, 1))(groups[0])
    group2 = Conv2D(filters=32, kernel_size=(3, 3))(groups[1])
    group3 = Conv2D(filters=64, kernel_size=(5, 5))(groups[2])
    
    # Concatenate the outputs from the three groups
    output = Concatenate()([group1, group2, group3])
    
    # Apply dropout to reduce overfitting
    dropout = Dropout(rate=0.2)(output)
    
    # Block 2
    # Define four branches that process the input separately
    branch1 = Conv2D(filters=16, kernel_size=(1, 1))(dropout)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3))(dropout)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5))(dropout)
    branch4 = MaxPooling2D(pool_size=(1, 1))(dropout)
    
    # Concatenate the outputs from all branches
    output = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Flatten the output
    flatten = Flatten()(output)
    
    # Fully connected layer
    output = Dense(units=128, activation='relu')(flatten)
    output = Dense(units=10, activation='softmax')(output)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model