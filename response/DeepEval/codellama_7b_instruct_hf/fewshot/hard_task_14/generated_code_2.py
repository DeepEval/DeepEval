import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten, Reshape, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(units=64, activation='relu')(main_path)
    main_path = Dense(units=32, activation='relu')(main_path)
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Flatten()(branch_path)
    
    # Combine main and branch paths
    combined_path = Add()([main_path, branch_path])
    
    # Fully connected layers
    combined_path = Dense(units=128, activation='relu')(combined_path)
    combined_path = Dense(units=64, activation='relu')(combined_path)
    combined_path = Dense(units=10, activation='softmax')(combined_path)
    
    # Define model
    model = keras.Model(inputs=input_layer, outputs=combined_path)
    
    return model