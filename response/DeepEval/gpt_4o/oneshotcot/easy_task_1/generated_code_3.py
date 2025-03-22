import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add first convolutional layer followed by max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    # Step 3: Add second convolutional layer followed by max pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    
    # Step 4: Add additional convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(max_pool2)
    
    # Step 5: Flatten the feature maps
    flatten_layer = Flatten()(conv3)
    
    # Step 6: Add first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 7: Add second fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Build and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model