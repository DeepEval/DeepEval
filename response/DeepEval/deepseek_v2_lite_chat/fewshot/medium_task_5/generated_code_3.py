import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main path
    block1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    block1 = MaxPooling2D()(block1)
    
    block2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(block1)
    branch_outputs = MaxPooling2D()(block2)
    
    # Branch path
    branch_inputs = inputs
    branch_outputs = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(branch_inputs)
    branch_outputs = MaxPooling2D()(branch_outputs)
    
    # Add main and branch paths
    combined_outputs = Add()([block2, branch_outputs])
    
    # Flatten and fully connected layers
    flat = Flatten()(combined_outputs)
    fc1 = Dense(units=128, activation='relu')(flat)
    fc2 = Dense(units=64, activation='relu')(fc1)
    output = Dense(units=10, activation='softmax')(fc2)
    
    # Model construction
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Create the model
model = dl_model()
model.summary()