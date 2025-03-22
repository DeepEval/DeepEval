from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    
    # Max-pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)
    
    # Add the output of the max-pooling layer with the input layer
    added = Add()([max_pooling, input_layer])
    
    # Flatten the output
    flatten_layer = Flatten()(added)
    
    # First fully connected layer
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    
    # Output layer with softmax activation for probability distribution across 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model