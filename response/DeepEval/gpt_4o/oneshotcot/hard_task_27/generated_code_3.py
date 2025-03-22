import keras
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Dense, Add, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 1: Extract spatial features with depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(input_layer)
    
    # Step 2: Layer normalization for training stability
    layer_norm = LayerNormalization()(depthwise_conv)
    
    # Step 3: Pass through two fully connected layers with the same number of channels as input
    flatten = Flatten()(layer_norm)
    fc1 = Dense(units=32*32*3, activation='relu')(flatten)
    fc2 = Dense(units=32*32*3, activation='relu')(fc1)
    
    # Step 4: Reshape back to original input shape for addition
    reshaped_fc2 = keras.layers.Reshape((32, 32, 3))(fc2)
    
    # Step 5: Combine original input with processed features
    added_output = Add()([input_layer, reshaped_fc2])
    
    # Step 6: Final classification using fully connected layers
    flatten_added = Flatten()(added_output)
    fc3 = Dense(units=512, activation='relu')(flatten_added)
    output_layer = Dense(units=10, activation='softmax')(fc3)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Initialize the model
model = dl_model()

# Display the model's architecture
model.summary()