import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First sequential block
    conv1 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    
    # Second sequential block
    conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Adding the input layer to the output of the second block
    resized_input = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    added = Add()([max_pool2, resized_input])
    
    # Flatten and fully connected layer
    flatten_layer = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model