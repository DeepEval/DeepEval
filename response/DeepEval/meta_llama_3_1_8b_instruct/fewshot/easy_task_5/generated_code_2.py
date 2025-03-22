import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    
    # Reduce dimensionality with a 1x1 convolution
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Extract features using a 3x3 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Restore dimensionality with a 1x1 convolution layer
    conv3 = Conv2DTranspose(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv2)
    
    # Flattening the output
    flatten_layer = Flatten()(conv3)
    
    # Fully connected layer with 10 neurons for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
print(model.summary())