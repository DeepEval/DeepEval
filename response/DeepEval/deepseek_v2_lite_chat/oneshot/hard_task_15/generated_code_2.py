import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Flatten, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_model = Conv2D(64, (3, 3), activation='relu')(input_layer)
    main_model = GlobalAveragePooling2D()(main_model)
    main_dense1 = Dense(256, activation='relu')(main_model)
    main_dense2 = Dense(128, activation='relu')(main_dense1)
    
    # Branch path
    branch_model = input_layer
    
    # Concatenate the outputs from both paths
    combined = Add()([main_dense2, branch_model])
    
    # Two additional fully connected layers
    combined_dense1 = Dense(256, activation='relu')(combined)
    combined_dense2 = Dense(128, activation='relu')(combined_dense1)
    
    # Output layer
    output_layer = Dense(10, activation='softmax')(combined_dense2)
    
    # Model construction
    model = keras.Model(inputs=[input_layer, branch_model], outputs=[output_layer])
    
    return model