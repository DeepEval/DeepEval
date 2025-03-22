import keras
from keras.layers import Input, Conv2D, ReLU, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    def main_path_block(input_tensor):
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)  
        return x
    
    for _ in range(3):
        x = main_path_block(x)

    # Branch Path
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Feature Fusion
    x = Concatenate()([x, branch_path]) 
    x = Add()([x, branch_path])

    x = BatchNormalization()(x)
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model