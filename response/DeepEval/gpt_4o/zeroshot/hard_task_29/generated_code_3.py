from tensorflow.keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Convolutional Block with a Main and a Branch Path
    # Main path
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # Branch path (direct connection)
    branch = input_layer
    
    # Combining the main path and branch path using addition
    combined = Add()([x, branch])
    
    # Block 2: Max Pooling with different scales
    # Pooling with 1x1
    pool_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(combined)
    pool_1x1 = Flatten()(pool_1x1)
    
    # Pooling with 2x2
    pool_2x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(combined)
    pool_2x2 = Flatten()(pool_2x2)
    
    # Pooling with 4x4
    pool_4x4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(combined)
    pool_4x4 = Flatten()(pool_4x4)
    
    # Concatenating the flattened outputs
    concatenated = Concatenate()([pool_1x1, pool_2x2, pool_4x4])
    
    # Fully connected layers for classification
    fc1 = Dense(units=128, activation='relu')(concatenated)
    output_layer = Dense(units=10, activation='softmax')(fc1)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Now, you can create the model instance
model = dl_model()