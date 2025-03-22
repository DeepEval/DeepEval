import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Concatenate, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    block1_output = Conv2D(filters=32, kernel_size=3, padding='same')(input_layer)
    block1_output = BatchNormalization()(block1_output)
    block1_output = ReLU()(block1_output)
    
    # Block 2
    block2_output = Conv2D(filters=64, kernel_size=3, padding='same')(input_layer)
    block2_output = BatchNormalization()(block2_output)
    block2_output = ReLU()(block2_output)
    
    # Block 3
    block3_output = Conv2D(filters=128, kernel_size=3, padding='same')(input_layer)
    block3_output = BatchNormalization()(block3_output)
    block3_output = ReLU()(block3_output)
    
    # Parallel branches
    branch1_output = Conv2D(filters=64, kernel_size=1, padding='same')(input_layer)
    branch2_output = Conv2D(filters=64, kernel_size=3, padding='same')(input_layer)
    
    # Concatenate all outputs
    concat_output = Concatenate()([block1_output, block2_output, block3_output, branch1_output, branch2_output])
    
    # Fully connected layers
    flatten_layer = Flatten()(concat_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])