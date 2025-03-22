import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Define the input layer with the shape of CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path: Two consecutive 3x3 convolutional layers followed by a max pooling layer
    main_path = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)
    
    # Branch path: A single 5x5 convolutional layer
    branch_path = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
    
    # Combine the outputs from both paths
    combined = Concatenate()([main_path, branch_path])
    
    # Flatten the combined output
    flatten_layer = Flatten()(combined)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model