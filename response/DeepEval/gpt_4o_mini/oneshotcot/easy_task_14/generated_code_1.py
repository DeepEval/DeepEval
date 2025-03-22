import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels
    
    # Global average pooling to compress the features
    gap = GlobalAveragePooling2D()(input_layer)
    
    # Two fully connected layers to generate weights
    fc1 = Dense(units=128, activation='relu')(gap)
    fc2 = Dense(units=32 * 32 * 3, activation='sigmoid')(fc1)  # Output size matches input shape
    
    # Reshape to match the input feature map dimensions
    reshaped_weights = Reshape((1, 1, 32, 32, 3))(fc2)
    
    # Element-wise multiplication with the input feature map
    multiplied_output = Multiply()([input_layer, reshaped_weights])
    
    # Flatten the result for the final output layer
    flatten_layer = Flatten()(multiplied_output)
    
    # Final fully connected layer to produce the output distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model