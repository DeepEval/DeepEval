import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, GlobalAveragePooling2D, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Feature extraction
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 2: Channel-wise refinement
    main_path = GlobalAveragePooling2D()(pool1)
    branch_path = Flatten()(main_path)
    adding_layer = Add()([main_path, branch_path])
    dense1 = Dense(64, activation='relu')(adding_layer)
    dense2 = Dense(32, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    # Define model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model