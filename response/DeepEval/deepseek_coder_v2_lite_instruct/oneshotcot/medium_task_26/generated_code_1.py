import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Step 2: Add convolutional layer to compress the input channels
    compressed = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Expand features through two parallel convolutional layers
    # 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(compressed)
    # 3x3 convolution
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(compressed)
    
    # Step 4: Concatenate the results
    expanded = Concatenate()([path1, path2])
    
    # Step 5: Add batch normalization
    batch_norm = BatchNormalization()(expanded)
    
    # Step 6: Flatten the result
    flatten_layer = Flatten()(batch_norm)
    
    # Step 7: Add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 8: Add another dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 9: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model