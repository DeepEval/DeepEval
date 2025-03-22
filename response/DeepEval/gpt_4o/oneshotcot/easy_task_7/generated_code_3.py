from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    # First Conv-Dropout block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    dropout1 = Dropout(rate=0.25)(conv1)
    
    # Second Conv-Dropout block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(dropout1)
    dropout2 = Dropout(rate=0.25)(conv2)
    
    # Restoring channels
    conv3 = Conv2D(filters=1, kernel_size=(3, 3), activation='relu', padding='same')(dropout2)
    
    # Branch path (direct connection to input)
    branch_path = input_layer
    
    # Combine paths using addition
    combined = Add()([conv3, branch_path])
    
    # Flatten and fully connected layer
    flatten = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model