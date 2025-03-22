import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():
    # Define the input layer with the shape of CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))
    
    # Global average pooling to compress spatial information
    global_avg_pooling = GlobalAveragePooling2D()(input_layer)

    # Two fully connected layers to generate weights the same size as the channels
    dense1 = Dense(units=32, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=3, activation='sigmoid')(dense1)  # '3' because of 3 color channels in input
    
    # Reshape to match the input channels
    channel_weights = Reshape((1, 1, 3))(dense2)
    
    # Multiply the input feature map with the generated weights
    scaled_feature_map = Multiply()([input_layer, channel_weights])
    
    # Flatten the result
    flatten_layer = Flatten()(scaled_feature_map)
    
    # Fully connected layer to produce the final classification output
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model