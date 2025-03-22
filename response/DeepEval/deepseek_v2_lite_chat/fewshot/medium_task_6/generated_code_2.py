import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, BatchNormalization, ReLU

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Initial convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation=ReLU(), padding='same')(inputs)
    
    # First block
    norm1 = BatchNormalization()(conv1)
    block1 = ReLU()(norm1)
    
    # Second block
    norm2 = BatchNormalization()(block1)
    block2 = ReLU()(norm2)
    
    # Third block
    norm3 = BatchNormalization()(block2)
    block3 = ReLU()(norm3)
    
    # Add the outputs of the blocks to enhance feature representation
    add_layer = Add()([conv1, block1, block2, block3])
    
    # Flatten and fully connected layers
    flatten = Flatten()(add_layer)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Print model summary
model.summary()