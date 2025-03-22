import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense

def dl_model():
    
    # Define the input layer with shape (224, 224, 3)
    input_layer = Input(shape=(224, 224, 3))
    
    # First feature extraction layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    
    # Second feature extraction layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Additional convolutional layers for dimensionality reduction
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    pool3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4)
    
    # Flatten the feature maps
    flatten = Flatten()(pool3)
    
    # First fully connected layer with dropout
    dense1 = Dense(units=512, activation='relu')(flatten)
    dropout1 = Dropout(0.2)(dense1)
    
    # Second fully connected layer with dropout
    dense2 = Dense(units=256, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    
    # Output layer with softmax activation
    output_layer = Dense(units=1000, activation='softmax')(dropout2)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model