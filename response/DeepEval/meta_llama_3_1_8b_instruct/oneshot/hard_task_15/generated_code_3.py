import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape, Multiply, Add
from keras import regularizers

def dl_model():
    
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Global average pooling to extract global information from the feature map
    global_avg_pooling = GlobalAveragePooling2D()(max_pooling)
    
    # Pass the pooled features through two fully connected layers to generate weights
    dense1 = Dense(units=128, activation='relu')(global_avg_pooling)
    weights = Dense(units=64, activation='relu')(dense1)
    
    # Reshape the weights to match the input layer's shape
    weights = Reshape((2, 2, 64))(weights)
    
    # Multiply the weights element-wise with the input feature map
    multiplied = Multiply()([max_pooling, weights])
    
    # Main path output
    main_path_output = multiplied
    
    # Branch path
    branch_path_output = input_layer
    
    # Add the outputs from both paths
    combined_output = Add()([main_path_output, branch_path_output])
    
    # Pass the combined output through two fully connected layers to produce the final probability distribution
    dense1 = Dense(units=128, activation='relu')(combined_output)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model