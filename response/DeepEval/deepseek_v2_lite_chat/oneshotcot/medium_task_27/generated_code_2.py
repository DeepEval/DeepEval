import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Dense, Flatten, Lambda

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional branch 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
    
    # Convolutional branch 2
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
    
    # Add branches
    add_layer = Add()([pool1, pool2])
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(add_layer)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Softmax attention layer
    attention_weights = Dense(units=2, activation='softmax', name='attention_weights')(dense1)
    
    # Multiply the outputs of each branch by their attention weights and sum
    weighted_output = Lambda(lambda tensors: keras.backend.sum(tensors[0] * attention_weights) + tensors[1])([dense2, add_layer])
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(weighted_output)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model