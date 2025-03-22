from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer for CIFAR-10 images (32x32 RGB)
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add the first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add the second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Step 4: Add max-pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Step 5: Directly add the max-pooling output with the input layer
    # We first need to adjust dimensions of input_layer to match max_pooling
    # Using a Conv2D layer with 1x1 kernel to adjust dimensions
    input_adjusted = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    # Add the features
    added_features = Add()([max_pooling, input_adjusted])
    
    # Step 6: Flatten the features
    flatten_layer = Flatten()(added_features)
    
    # Step 7: Add the first dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 8: Add the second dense layer (output layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model