import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Concatenate, Dropout, Reshape

# Constants
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10

def dl_model():
    # Input Layer
    inputs = Input(shape=INPUT_SHAPE)
    
    # Block 1
    block1 = inputs
    for _ in range(3):
        block1 = AveragePooling2D(pool_size=(1, 1))(block1)  # 1x1
        block1 = AveragePooling2D(pool_size=(2, 2))(block1)  # 2x2
        block1 = AveragePooling2D(pool_size=(4, 4))(block1)  # 4x4
    
    # Flatten and Dropout
    flat1 = Flatten()(block1)
    block1_dropout = Dropout(0.5)(flat1)
    
    # Concatenate for Block 1 output
    block1_output = block1_dropout
    
    # Block 2
    block2_input = block1_output
    for _ in range(4):
        if _ == 0:
            block2_input = Conv2D(64, (1, 1), activation='relu')(block2_input)
        elif _ == 1:
            block2_input = Conv2D(64, (3, 3), activation='relu')(block2_input)
        elif _ == 2:
            block2_input = Conv2D(64, (3, 3), activation='relu')(block2_input)
        else:
            block2_input = MaxPooling2D(pool_size=(2, 2))(block2_input)
    
    # Flatten and Dropout for Block 2
    flat2 = Flatten()(block2_input)
    block2_dropout = Dropout(0.5)(flat2)
    
    # Fully Connected Layer
    outputs = Dense(NUM_CLASSES, activation='softmax')(block2_dropout)
    
    # Model
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()