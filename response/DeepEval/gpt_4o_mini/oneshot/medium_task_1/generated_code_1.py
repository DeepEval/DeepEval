import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)
    
    # Adding the input layer with the max pooling output (skip connection)
    added_output = Add()([input_layer, max_pooling])  # Input layer must have the same shape as max_pooling
    
    # Flattening the output
    flatten_layer = Flatten()(added_output)
    
    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# You can create an instance of the model by calling the function
model = dl_model()