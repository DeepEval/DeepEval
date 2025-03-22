import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path block
    def block_main(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        relu1 = Activation('relu')(bn1)
        pool1 = AveragePooling2D(pool_size=(2, 2))(relu1)
        
        return pool1
    
    # Branch path block
    def block_branch(input_tensor):
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        bn2 = BatchNormalization()(conv2)
        relu2 = Activation('relu')(bn2)
        return relu2
    
    # Main path outputs
    main_path_output = block_main(input_tensor=input_layer)
    
    # Branch path outputs
    branch_path_output = block_branch(input_tensor=input_layer)
    
    # Feature fusion
    fused_output = Add()([main_path_output, branch_path_output])
    
    # Additional layers
    flattened = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.summary()