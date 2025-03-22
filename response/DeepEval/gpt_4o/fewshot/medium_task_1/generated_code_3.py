import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    
    # Max pooling layer
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Add the pooled features to the input layer (broadcasting the input to match pool shape)
    added_features = Add()([input_layer, pool])
    
    # Flatten the features
    flatten = Flatten()(added_features)
    
    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten)
    
    # Second fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model