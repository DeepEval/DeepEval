import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate, Reshape
from keras.layers import InputSpec

def dl_model():
    # Define input specification
    inputs = Input(shape=(32, 32, 3), name='input')
    input_spec = InputSpec(shape=inputs.shape)
    
    # Branch 1: Global Average Pooling, two fully connected layers
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    branch1 = GlobalAveragePooling2D()(branch1)
    branch1 = Dense(512, activation='relu')(branch1)
    branch1 = Dense(256, activation='relu')(branch1)
    branch1 = Reshape((64,))(branch1)
    
    # Branch 2: Global Average Pooling, two fully connected layers
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    branch2 = GlobalAveragePooling2D()(branch2)
    branch2 = Dense(512, activation='relu')(branch2)
    branch2 = Dense(256, activation='relu')(branch2)
    branch2 = Reshape((64,))(branch2)
    
    # Concatenate outputs from both branches
    outputs = concatenate([branch1, branch2])
    
    # Flatten and fully connected layer for classification
    outputs = Flatten()(outputs)
    final_output = Dense(10, activation='softmax')(outputs)
    
    # Define the model
    model = Model(inputs=inputs, outputs=final_output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
# model = dl_model()
# model.summary()