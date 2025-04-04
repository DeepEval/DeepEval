import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Step 3: Add max pooling layer
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    # Step 4: Add additional convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling1)
    
    # Step 5: Flatten the result
    flatten_layer = Flatten()(conv2)
    
    # Step 6: Add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 7: Add dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 8: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Step 9: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model