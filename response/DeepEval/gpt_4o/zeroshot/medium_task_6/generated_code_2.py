from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense, Softmax
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Convolution
    initial_conv = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    
    # Block 1
    x1 = Conv2D(32, (3, 3), padding='same')(initial_conv)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    
    # Block 2
    x2 = Conv2D(32, (3, 3), padding='same')(initial_conv)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    
    # Block 3
    x3 = Conv2D(32, (3, 3), padding='same')(initial_conv)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)
    
    # Add outputs of the blocks with the initial convolution's output
    added = Add()([initial_conv, x1, x2, x3])
    
    # Flatten and fully connected layers
    x = Flatten()(added)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    # Output layer with Softmax
    output_layer = Dense(10, activation='softmax')(x)
    
    # Construct model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
# model = dl_model()
# model.summary()