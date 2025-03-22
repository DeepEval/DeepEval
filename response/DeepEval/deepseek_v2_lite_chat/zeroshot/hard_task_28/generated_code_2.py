from keras.models import Model
from keras.layers import Input, Conv2D, LayerNormalization, Conv2DTranspose, Add, Flatten, Dense, concatenate
from keras.layers import ReLU

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    x = Conv2D(64, (7, 7), padding='same', activation='relu')(input_layer)
    x = LayerNormalization()(x)
    x = Conv2D(64, (1, 1), padding='same')(x)
    x = ReLU()(x)
    
    x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    x = Conv2D(64, (1, 1), padding='same')(x)
    x = ReLU()(x)
    
    # Branch path
    y = input_layer
    
    # Combine the outputs through addition
    combined = Add()([x, y])
    
    # Flatten and process through fully connected layers
    combined = Flatten()(combined)
    output = Dense(10, activation='softmax')(combined)  # Assuming 10 classes for CIFAR-10
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Example usage:
model = dl_model()
model.summary()