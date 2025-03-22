import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Step 3: Add first max pooling layer
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    # Step 4: Add second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pool1)
    
    # Step 5: Add second max pooling layer
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    
    # Step 6: Add an additional convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pool2)
    
    # Step 7: Flatten layer
    flatten_layer = Flatten()(conv3)
    
    # Step 8: Add first dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 9: Add output dense layer
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model