import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(x):
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return x
    
    # Branch path
    def branch_path(x):
        x = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(x)
        return x
    
    # Apply main path and branch path
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    
    # Add main and branch outputs
    combined = Add()([main_output, branch_output])
    
    # Flatten and fully connected layers
    flatting = Flatten()(combined)
    fc = Dense(units=128, activation='relu')(flatting)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(fc)
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Print model summary
model.summary()