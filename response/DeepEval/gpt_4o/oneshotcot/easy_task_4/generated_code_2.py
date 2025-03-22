import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Define the first block
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1_1)
    conv1_3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1_2)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1_3)
    
    # Step 3: Define the second block
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(max_pool1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2_1)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2_2)
    conv2_4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2_3)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2_4)
    
    # Step 4: Flatten the feature maps
    flatten_layer = Flatten()(max_pool2)
    
    # Step 5: Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Step 6: Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model