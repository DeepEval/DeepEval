import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
        pool = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        return pool
    
    main_pool = main_path(input_tensor=input_layer)
    
    # Branch path
    def branch_path(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=512, activation='relu')(avg_pool)
        dense2 = Dense(units=256, activation='relu')(dense1)
        channel_weights = Dense(units=3, activation='softmax')(dense2)  # Generate channel weights
        
        # Reshape channel weights
        channel_weights = keras.layers.Reshape((1, 1, 3))(channel_weights)
        
        # Multiply input with channel weights
        output = keras.layers.Multiply()([input_tensor, keras.layers.Lambda(lambda x: K.expand_dims(x))([channel_weights])])
        
        return output
    
    branch_output = branch_path(input_tensor=main_pool)
    
    # Concatenate outputs from main path and branch path
    combined = Concatenate()([branch_output, main_pool])
    
    # Additional convolutional layers
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(combined)
    conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(conv4)
    
    # Fully connected layers for classification
    dense3 = Dense(units=512, activation='relu')(conv5)
    dense4 = Dense(units=10, activation='softmax')(dense3)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=dense4)
    
    return model

# Create the model
model = dl_model()
model.summary()