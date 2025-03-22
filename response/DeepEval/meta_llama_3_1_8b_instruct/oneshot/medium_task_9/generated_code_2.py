import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_layer)
    
    # Basic block
    def basic_block(input_tensor):
        branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        branch = BatchNormalization()(branch)
        branch = ReLU()(branch)
        
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = ReLU()(main_path)
        
        output_tensor = Add()([main_path, branch])
        output_tensor = ReLU()(output_tensor)
        
        return output_tensor
    
    block1 = basic_block(conv)
    block2 = basic_block(block1)
    
    # Feature extraction in the branch
    branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=None)(block2)
    
    # Feature fusion
    output_tensor = Add()([block2, branch])
    output_tensor = ReLU()(output_tensor)
    
    # Average pooling and flattening
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(output_tensor)
    avg_pool = Flatten()(avg_pool)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(avg_pool)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model