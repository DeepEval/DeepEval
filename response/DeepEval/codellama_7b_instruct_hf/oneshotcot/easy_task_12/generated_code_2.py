import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Define the main path
    main_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)
    
    # Define the branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch_path = Conv2D(32, (3, 3), activation='relu')(branch_path)
    branch_path = MaxPooling2D((2, 2))(branch_path)
    
    # Define the output layer
    output_layer = Concatenate()([main_path, branch_path])
    output_layer = Flatten()(output_layer)
    output_layer = Dense(128, activation='relu')(output_layer)
    output_layer = Dense(10, activation='softmax')(output_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model