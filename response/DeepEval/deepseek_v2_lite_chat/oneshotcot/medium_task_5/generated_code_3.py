import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def block(input_tensor):
        # Block 1
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
        
        # Block 2
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
        
        return maxpool2
    
    main_output = block(input_tensor=input_layer)
    
    # Branch path
    def branch_block(input_tensor):
        # Block 1
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)
        return maxpool
    
    branch_output = branch_block(input_tensor=input_layer)
    
    # Combine paths
    combined_output = Add()([main_output, branch_output])
    
    # Flatten and dense layers
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()