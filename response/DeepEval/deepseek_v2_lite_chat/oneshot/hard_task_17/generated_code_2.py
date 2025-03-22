import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Feature extraction using global average pooling and fully connected layers
    avg_pool = GlobalAveragePooling2D()(input_layer)
    fc1 = Dense(units=128, activation='relu')(avg_pool)
    fc2 = Dense(units=64, activation='relu')(fc1)
    
    # Block 2: Deep feature extraction using convolutional layers and max pooling
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Branch connection from Block 1 to Block 2
    branch_output = Dense(units=64, activation='relu')(fc1)
    
    # Concatenate the outputs from Block 1 and Block 2
    concat = Concatenate()([max_pool, branch_output])
    
    # Additional dense layers
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()