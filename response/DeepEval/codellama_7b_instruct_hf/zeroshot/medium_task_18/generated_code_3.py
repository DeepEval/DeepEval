import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Load the VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add a new input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Add a new 1x1 convolution layer
    conv1 = Conv2D(32, (1, 1), padding='same')(input_layer)
    
    # Add a new 3x3 convolution layer
    conv2 = Conv2D(32, (3, 3), padding='same')(conv1)
    
    # Add a new 5x5 convolution layer
    conv3 = Conv2D(32, (5, 5), padding='same')(conv2)
    
    # Add a new 3x3 max pooling layer
    pool1 = MaxPooling2D((3, 3), strides=(2, 2))(conv3)
    
    # Add a new flattening layer
    flatten1 = Flatten()(pool1)
    
    # Add a new fully connected layer
    dense1 = Dense(128, activation='relu')(flatten1)
    
    # Add a new fully connected layer
    dense2 = Dense(10, activation='softmax')(dense1)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=dense2)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model