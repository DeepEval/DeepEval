import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Reshape, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(128, activation='relu')(main_path)
    main_path = Dense(64, activation='relu')(main_path)
    main_path_weights = Dense(32, activation='relu')(main_path)
    main_path_weights = Reshape((1, 1, 32))(main_path_weights)
    main_path_weights = Multiply()([input_layer, main_path_weights])
    
    # Branch path
    branch_path = input_layer
    
    # Combine main and branch paths
    combined = Add()([main_path_weights, branch_path])
    
    # Additional fully connected layers
    combined = Conv2D(32, (3, 3), activation='relu')(combined)
    combined = Conv2D(64, (3, 3), activation='relu')(combined)
    combined = GlobalAveragePooling2D()(combined)
    output_layer = Dense(10, activation='softmax')(combined)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model