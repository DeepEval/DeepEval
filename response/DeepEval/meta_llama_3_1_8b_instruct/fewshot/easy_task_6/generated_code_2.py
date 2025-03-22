import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch path
    branch_path = input_layer
    
    # Combine the two paths
    adding_layer = Add()([conv2, branch_path])
    
    # Flatten the output
    flatten_layer = Flatten()(adding_layer)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model