import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, Flatten, Dense

def dl_model():
    # Define the main path
    input_layer = Input(shape=(28, 28, 1))
    
    # First convolution block in the main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(0.2)(conv1)
    
    # Second convolution block in the main path
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(conv2)
    
    # Restoring the number of channels
    conv3 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout2)
    
    # Define the branch path
    branch_conv = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs from both paths
    added = Add()([conv3, branch_conv])
    
    # Flatten the result
    flatten_layer = Flatten()(added)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model