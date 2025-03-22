import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers.experimental.preprocessing import ImageScaler, RandomCrop, RandomFlip

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = Conv2D(32, (1, 1), activation='relu')(input_layer)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = Conv2D(32, (5, 5), activation='relu')(main_path)
    main_path = Concatenate()([main_path, main_path, main_path])
    
    # Branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch_path = RandomCrop(32)(branch_path)
    branch_path = RandomFlip(mode='horizontal')(branch_path)
    
    # Concatenate main and branch paths
    merged_path = Concatenate()([main_path, branch_path])
    
    # Batch normalization and flatten
    merged_path = BatchNormalization()(merged_path)
    merged_path = Flatten()(merged_path)
    
    # Dense layers
    merged_path = Dense(256, activation='relu')(merged_path)
    merged_path = Dense(128, activation='relu')(merged_path)
    output_layer = Dense(10, activation='softmax')(merged_path)
    
    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model