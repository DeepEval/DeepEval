import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # CIFAR-10 images are 32x32 with 3 color channels
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    
    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
    
    # Adding the max pooling output with the input layer
    # Adjusting dimensions if necessary to perform the addition
    added_features = Add()([max_pooling, input_layer])
    
    # Flatten the output
    flatten_layer = Flatten()(added_features)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model