import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Two 3x3 convolutions, average pooling, and addition
    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
        avg_pool = MaxPooling2D(pool_size=(2, 2))(conv2)
        add_layer = Add()([input_tensor, avg_pool])
        return add_layer
    
    # Block 2: Global average pooling, two fully connected layers, and reshaping
    def block2(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        channel_weights = Dense(units=512, activation='relu')(avg_pool)
        channel_weights = Dense(units=256, activation='relu')(channel_weights)
        channel_weights = Dense(units=128, activation='relu')(channel_weights)
        reshape_layer = Flatten()(channel_weights)
        multiply_layer = Dense(units=128)(reshape_layer)
        multiply_layer = keras.layers.multiply([input_tensor, multiply_layer])
        output_layer = Dense(units=10, activation='softmax')(multiply_layer)
        return output_layer
    
    # Construct the model
    add_layer = block1(input_layer)
    output_layer = block2(add_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model