import keras
from keras.layers import Input, Dense, Flatten, Reshape, Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Block 1: Feature extraction
    block1_output = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    block1_output = MaxPooling2D(pool_size=(2, 2))(block1_output)
    block1_output = Flatten()(block1_output)
    
    # FC layers
    dense1 = Dense(units=128, activation='relu')(block1_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Reshape to match Block 2 input shape
    reshaped = Reshape((1024,))(dense2)
    
    # Block 2: Deep feature extraction
    block2_output = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    block2_output = MaxPooling2D(pool_size=(2, 2))(block2_output)
    branch_output = GlobalAveragePooling2D()(block2_output)
    
    # Fusion of outputs from main path and branch
    fused_output = Concatenate()([branch_output, dense2])
    
    # Final dense layers for classification
    dense3 = Dense(units=128, activation='relu')(fused_output)
    output = Dense(units=10, activation='softmax')(dense3)
    
    # Model construction
    model = keras.Model(inputs=inputs, outputs=output)
    
    return model

# Instantiate and return the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())