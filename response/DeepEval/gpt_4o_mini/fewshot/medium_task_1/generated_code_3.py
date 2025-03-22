import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Input layer for CIFAR-10 images (32x32 pixels, 3 color channels)
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Max pooling layer
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Adding the pooled features with the input layer
    added_features = Add()([pool, input_layer])
    
    # Flatten the combined features
    flatten_layer = Flatten()(added_features)
    
    # Fully connected layer with 128 units
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Fully connected layer with 64 units
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer with 10 units (for CIFAR-10 classes) and softmax activation
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model