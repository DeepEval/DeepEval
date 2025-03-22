import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Input layer for CIFAR-10 images (32x32 with 3 color channels)
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)
    
    # Adding the output of the max pooling layer with the input
    added_features = Add()([max_pooling, input_layer])
    
    # Flatten layer to convert 3D feature maps to 1D feature vectors
    flatten_layer = Flatten()(added_features)
    
    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Second fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model