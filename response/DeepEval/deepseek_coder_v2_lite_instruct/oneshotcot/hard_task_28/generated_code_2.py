import keras
from keras.layers import Input, Conv2D, LayerNormalization, Add, Flatten, Dense, DepthwiseConv2D
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = DepthwiseConv2D(kernel_size=(7, 7), padding='same', depth_multiplier=1)(input_layer)
    main_path = LayerNormalization()(main_path)  # Layer normalization for standardization
    main_path = Conv2D(filters=input_layer.shape[3], kernel_size=(1, 1))(main_path)  # 1x1 pointwise convolution
    main_path = Conv2D(filters=input_layer.shape[3], kernel_size=(1, 1))(main_path)  # Another 1x1 pointwise convolution
    
    # Branch path
    branch_path = input_layer  # Directly connects to the input
    
    # Combine both paths
    combined_path = Add()([main_path, branch_path])
    
    # Flatten the combined output
    flattened_layer = Flatten()(combined_path)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flattened_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and return the model
model = dl_model()
model.summary()