import keras
from keras.layers import Input, Conv2D, LayerNormalization, Dense, Add, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # Depthwise convolution
        depthwise = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', depth_axis=1, activation='relu')(input_tensor)
        # Layer normalization
        layer_norm = LayerNormalization()(depthwise)
        # Pointwise convolution 1
        pointwise1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(layer_norm)
        # Pointwise convolution 2
        pointwise2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pointwise1)
        
        return pointwise2

    main_output = main_path(input_tensor=input_layer)
    
    # Branch path
    branch_output = input_layer
    
    # Combine outputs
    combined_output = Add()([main_output, branch_output])
    
    # Flatten and fully connected layers
    flatten = Flatten()(combined_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()