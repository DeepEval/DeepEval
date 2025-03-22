import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_output = GlobalAveragePooling2D()(input_layer)
    main_output = Dense(units=1024, activation='relu')(main_output)
    main_output = Dense(units=512, activation='relu')(main_output)
    
    # Branch path
    branch_output = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_layer)
    branch_output = GlobalAveragePooling2D()(branch_output)
    
    # Add main and branch outputs
    combined_output = Concatenate()([main_output, branch_output])
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(combined_output)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Print model summary
model.summary()