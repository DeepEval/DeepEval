import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Apply global average pooling
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    
    # Step 3: First fully connected layer for channel-wise weights
    dense1 = Dense(units=32*3, activation='relu')(global_avg_pool)  # Assuming input channels are 32*3 for this layer
    
    # Step 4: Second fully connected layer for channel-wise weights
    dense2 = Dense(units=32*3, activation='relu')(dense1)  # Output size matches the input channels
    
    # Step 5: Reshape the weights to align with the input shape
    weights_reshaped = Reshape((1, 1, 32*3))(dense2)
    
    # Step 6: Element-wise multiplication with the input feature map
    scaled_features = Multiply()([input_layer, weights_reshaped])
    
    # Step 7: Flatten the result
    flatten_layer = Flatten()(scaled_features)
    
    # Step 8: Final fully connected layer to obtain the classification output
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes
    
    # Step 9: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model