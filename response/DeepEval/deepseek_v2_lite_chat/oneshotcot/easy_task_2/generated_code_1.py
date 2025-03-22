import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def dl_model():
    # Input layer
    input_layer = Input(shape=(224, 224, 3))
    
    # Feature extraction layers
    def fe(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv1)
        return pool1
    
    fe_output1 = fe(input_tensor=input_layer)
    fe_output2 = fe(input_tensor=input_layer)
    
    # Additional convolutional layers and average pooling
    def additional_conv(input_tensor):
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv3)
        return pool2
    
    additional_conv_output = additional_conv(fe_output2)
    
    # Flattening and fully connected layers
    flat = Flatten()(additional_conv_output)
    
    fc1 = Dense(units=256, activation='relu')(flat)
    dropout1 = LeakyReLU(alpha=0.1)(fc1)
    
    fc2 = Dense(units=128, activation='relu')(dropout1)
    dropout2 = LeakyReLU(alpha=0.1)(fc2)
    
    output_layer = Dense(units=1000, activation='softmax')(dropout2)  # Assuming 1000 classes
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model