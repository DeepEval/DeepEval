import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)
    main_path = GlobalAveragePooling2D()(main_path)
    
    # Branch path
    branch_input = input_layer
    branch_path = branch_input  # No modifications on this branch
    
    # Concatenate the outputs of both paths
    combined_output = Concatenate()([main_path, branch_path])
    
    # Fully connected layers
    fc1 = Dense(units=128, activation='relu')(combined_output)
    fc2 = Dense(units=64, activation='relu')(fc1)
    output_layer = Dense(units=10, activation='softmax')(fc2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model