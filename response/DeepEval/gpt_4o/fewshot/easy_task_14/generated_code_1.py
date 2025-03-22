import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Reshape, Flatten

def dl_model():
    # Define input layer with shape matching CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))
    
    # Global average pooling to compress input features
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    
    # First fully connected layer to generate intermediate channel-wise weights
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    
    # Second fully connected layer to produce final channel-wise weights
    dense2 = Dense(units=3, activation='sigmoid')(dense1)
    
    # Reshape weights to align with input shape
    weights = Reshape((1, 1, 3))(dense2)
    
    # Element-wise multiplication of input features with weights
    scaled_features = Multiply()([input_layer, weights])
    
    # Flatten the result for final classification
    flatten = Flatten()(scaled_features)
    
    # Final fully connected layer to produce probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model