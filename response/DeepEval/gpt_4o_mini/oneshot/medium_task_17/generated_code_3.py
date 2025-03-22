import keras
from keras.layers import Input, Permute, Reshape, Dense
from keras.models import Model

def dl_model():
    # Define the input shape for CIFAR-10 images (32, 32, 3)
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape the input to (height, width, groups, channels_per_group)
    # Here, groups = 3 and channels_per_group = 1
    reshaped = Reshape((32, 32, 3, 1))(input_layer)
    
    # Permute dimensions to swap the last two dimensions
    # This will change the shape from (32, 32, 3, 1) to (32, 32, 1, 3)
    permuted = Permute((0, 1, 3, 2))(reshaped)
    
    # Reshape back to original input shape (32, 32, 3)
    reshaped_back = Reshape((32, 32, 3))(permuted)
    
    # Flatten the reshaped tensor and add a fully connected layer for classification
    flatten_layer = Flatten()(reshaped_back)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model