import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Initial Convolutional Layer
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the basic block
    def basic_block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        return x

    # First Basic Block
    branch1 = basic_block(x)
    branch2 = basic_block(x)

    # Feature Fusion
    x = Add()([branch1, branch2])

    # Second Basic Block
    branch1 = basic_block(x)
    branch2 = basic_block(x)

    # Feature Fusion
    x = Add()([branch1, branch2])

    # Branch Feature Extraction
    branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # Combine Branch and Main Path
    x = Add()([x, branch])

    # Average Pooling
    x = AveragePooling2D(pool_size=(8, 8))(x)

    # Flatten and Fully Connected Layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model