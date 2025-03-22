from keras.layers import Input, Conv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    conv1_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_main = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_main)
    
    # Branch path
    branch_path = input_layer  # Direct connection to the input
    
    # Combine paths through addition
    combined = Add()([conv2_main, branch_path])
    
    # Flatten and fully connected layer
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model