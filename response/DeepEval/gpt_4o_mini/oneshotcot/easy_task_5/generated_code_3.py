import keras
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model

def dl_model():
    
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels and single channel (grayscale)

    # Step 2: Reduce input dimensionality with a 1x1 convolution
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Step 3: Extract features using a 3x3 convolutional layer
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    
    # Step 4: Restore dimensionality with another 1x1 convolution layer
    restore_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv3x3)
    
    # Step 5: Flatten the output
    flatten_layer = Flatten()(restore_conv)
    
    # Step 6: Fully connected layer with 10 neurons for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for MNIST

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model