import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Add, BatchNormalization, ReLU, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation=ReLU, padding='same')(input_layer)
    
    # Batch normalization and ReLU activation for the main path
    bn1 = BatchNormalization()(conv1)
    
    # Branch path with a direct connection to the block input
    branch_input = conv1
    
    # Second convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation=ReLU, padding='same')(bn1)
    
    # Batch normalization and ReLU activation for the main path
    bn2 = BatchNormalization()(conv2)
    
    # Basic block for the main path
    main_path = bn2
    
    # Basic block for the branch path
    branch_path = branch_input
    
    # Add main path and branch path
    add_layer = Add()([main_path, branch_path])
    
    # Average pooling layer
    avg_pool = AveragePooling2D(pool_size=(2, 2))(add_layer)
    
    # Flatten layer
    flatten = Flatten()(avg_pool)
    
    # Fully connected layer
    dense = Dense(units=512, activation='relu')(flatten)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()