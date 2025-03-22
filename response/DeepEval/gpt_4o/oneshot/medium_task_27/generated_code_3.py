import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Softmax

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch with 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second branch with 5x5 convolution
    branch2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine branches using addition
    combined = Add()([branch1, branch2])
    
    # Global Average Pooling
    global_avg_pooling = GlobalAveragePooling2D()(combined)
    
    # Fully connected layers to produce attention weights
    fc1 = Dense(units=64, activation='relu')(global_avg_pooling)
    fc2 = Dense(units=2, activation='softmax')(fc1)
    
    # Split attention weights
    weight1, weight2 = fc2[:, 0], fc2[:, 1]
    
    # Reshape weights for multiplication
    weight1 = Multiply()([branch1, weight1])
    weight2 = Multiply()([branch2, weight2])
    
    # Final weighted output by adding the weighted branches
    weighted_output = Add()([weight1, weight2])
    
    # Output layer with probability distribution over 10 classes
    output_layer = Dense(units=10, activation='softmax')(weighted_output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model