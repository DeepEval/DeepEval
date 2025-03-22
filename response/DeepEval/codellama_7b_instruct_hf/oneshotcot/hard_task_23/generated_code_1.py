import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    # Load the VGG-16 model
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    
    # Freeze the VGG-16 layers
    for layer in vgg.layers:
        layer.trainable = False
    
    # Add a new input layer for the model
    input_layer = Input(shape=(32, 32, 3))
    
    # Add a 1x1 convolutional layer
    conv1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    # Add three branches
    branch1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    branch3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch2)
    branch3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch3)
    
    # Concatenate the branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Add a batch normalization layer
    batch_norm = BatchNormalization()(concatenated)
    
    # Add a flatten layer
    flattened = Flatten()(batch_norm)
    
    # Add a fully connected layer
    output_layer = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model