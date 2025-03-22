import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Feature extraction
    block1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    block1 = MaxPooling2D(pool_size=(2, 2))(block1)
    block1 = GlobalAveragePooling2D()(block1)
    
    # Block 2: Deep feature extraction
    branch = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    branch = MaxPooling2D(pool_size=(2, 2))(branch)
    
    # Concatenate outputs from Block 1 and Block 2
    x = Add()([block1, branch])
    
    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])