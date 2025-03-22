import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial Convolutional Layer
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Basic Block
    def basic_block(input_tensor):
        x1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x1 = BatchNormalization()(x1)
        x1 = Add()([x1, input_tensor])  
        return x1

    # First Block
    x = basic_block(x)
    
    # Branch Extraction
    branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # Second Block
    x = basic_block(x)

    # Feature Fusion
    x = Add()([x, branch])

    # Average Pooling
    x = AveragePooling2D(pool_size=(8, 8), strides=(8, 8))(x)

    # Flatten and Fully Connected Layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model