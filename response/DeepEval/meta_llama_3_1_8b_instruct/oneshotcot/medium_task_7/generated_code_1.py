import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():
    
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add convolutional layer 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Step 3: Add convolutional layer 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Step 4: Add convolutional layer 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Step 5: Add convolutional layer 4 directly
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 6: Add the outputs of the first two convolutional layers to the output of the third convolutional layer
    added_conv3 = Add()([conv1, conv2, conv3])
    
    # Step 7: Add the output of the fourth convolutional layer to the output of the previous step
    added_conv4 = Add()([added_conv3, conv4])
    
    # Step 8: Add batch normalization layer
    bath_norm = BatchNormalization()(added_conv4)
    
    # Step 9: Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # Step 10: Add dense layer 1
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 11: Add dense layer 2
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 12: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Step 13: Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model