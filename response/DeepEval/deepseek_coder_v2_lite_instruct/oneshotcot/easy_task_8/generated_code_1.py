import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add depthwise separable convolutional layer
    conv1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_conv_kernel=(3, 3), depthwise_padding='same')(input_layer)
    
    # Step 3: Add 1x1 convolutional layer for feature extraction
    conv1_1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Step 4: Add dropout layer to mitigate overfitting
    dropout1 = Dropout(0.25)(conv1_1)
    
    # Step 5: Add another depthwise separable convolutional layer
    conv2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_conv_kernel=(3, 3), depthwise_padding='same')(dropout1)
    
    # Step 6: Add another 1x1 convolutional layer for feature extraction
    conv2_1 = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Step 7: Add dropout layer to mitigate overfitting
    dropout2 = Dropout(0.25)(conv2_1)
    
    # Step 8: Flatten the result
    flatten_layer = Flatten()(dropout2)
    
    # Step 9: Add fully connected layer to generate classification probabilities
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Step 10: Build the model
    model = Model(inputs=input_layer, outputs=dense_layer)
    
    return model