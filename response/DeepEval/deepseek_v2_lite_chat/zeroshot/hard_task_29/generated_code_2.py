from keras.models import Model
from keras.layers import Input, Conv2D, Add, Concatenate, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Main path
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Branch path
    branch_output = Conv2D(32, (3, 3), activation='relu')(inputs)
    
    # Combine outputs from main path and branch path
    combined_output = Add()([x, branch_output])
    
    # Second block
    y = Conv2D(64, (3, 3), activation='relu')(combined_output)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y)
    y = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y)
    
    # Flatten and fully connected layers
    y = Flatten()(y)
    y = Dense(64, activation='relu')(y)
    outputs = Dense(10, activation='softmax')(y)  # Assuming 10 classes for MNIST
    
    # Model construction
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()