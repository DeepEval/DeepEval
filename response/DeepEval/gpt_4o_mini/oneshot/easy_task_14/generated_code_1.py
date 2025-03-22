import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Reshape, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels
    global_avg_pool = GlobalAveragePooling2D()(input_layer)  # Global Average Pooling to capture global information
    
    # Two fully connected layers to generate weights
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=3, activation='sigmoid')(dense1)  # Generate weights corresponding to the input channels

    # Reshaping weights to match the input feature map's channels
    reshaped_weights = Reshape((1, 1, 3))(dense2)
    
    # Multiply the input feature map with the learned weights
    scaled_output = Multiply()([input_layer, reshaped_weights])
    
    # Flatten the result and pass through a fully connected layer for classification
    flatten_layer = Flatten()(scaled_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model