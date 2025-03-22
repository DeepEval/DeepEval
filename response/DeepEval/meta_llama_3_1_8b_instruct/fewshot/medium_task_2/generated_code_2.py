import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    
    # Define the input layer with shape (32, 32, 3) for the CIFAR-10 dataset
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path: two consecutive 3x3 convolutional layers followed by a max pooling layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Branch path: a single 5x5 convolutional layer that also processes the input
    branch_conv = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the features from both paths
    combined = Concatenate()([max_pool, branch_conv])
    
    # Flatten the combined features
    flatten_layer = Flatten()(combined)
    
    # Map the flattened features to a probability distribution across 10 classes using two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model