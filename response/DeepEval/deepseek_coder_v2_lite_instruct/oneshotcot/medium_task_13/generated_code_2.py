import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add first max pooling layer
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Step 4: Add second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    
    # Step 5: Add second max pooling layer
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Step 6: Add third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool2)
    
    # Step 7: Concatenate the outputs of the convolutional layers along the channel dimension
    concatenated = Concatenate(axis=-1)([input_layer, conv1, conv2, conv3])
    
    # Step 8: Add batch normalization
    batch_norm = BatchNormalization()(concatenated)
    
    # Step 9: Flatten the result
    flatten_layer = Flatten()(batch_norm)
    
    # Step 10: Add first dense layer
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    
    # Step 11: Add second dense layer
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Step 12: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model