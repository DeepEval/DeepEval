import keras
from keras.layers import Input, Conv2D, Add, Dense, Flatten

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add sequential convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    
    # Step 3: Add separate convolutional layer processing the input directly
    conv_direct = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Step 4: Add outputs of the first two convolutional layers with the output of the third convolutional layer
    add_conv_layers = Add()([conv1, conv2, conv3])
    
    # Step 5: Add outputs from all paths
    add_all_paths = Add()([add_conv_layers, conv_direct])
    
    # Step 6: Add two fully connected layers for classification
    flatten_layer = Flatten()(add_all_paths)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model