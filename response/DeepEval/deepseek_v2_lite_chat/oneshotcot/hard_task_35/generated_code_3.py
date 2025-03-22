import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Flatten, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1
    # Block 1
    block1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    block1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(block1)
    block1 = GlobalAveragePooling2D()(block1)
    
    # Dense layer 1
    dense1 = Dense(units=512, activation='relu')(block1)
    
    # Branch 2
    # Block 2
    block2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(block2)
    block2 = GlobalAveragePooling2D()(block2)
    
    # Dense layer 2
    dense2 = Dense(units=512, activation='relu')(block2)
    
    # Concatenate outputs from both branches
    concat = Concatenate()([dense1, dense2])
    
    # Dense layer for final classification
    output_layer = Dense(units=10, activation='softmax')(concat)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()