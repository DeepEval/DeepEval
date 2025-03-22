import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():
    # CIFAR-10 images are 32x32 with 3 channels
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 1: Reshape the input to (height, width, groups, channels_per_group)
    groups = 3
    channels_per_group = 3 // groups  # Since we have 3 channels in CIFAR-10
    reshaped = Reshape((32, 32, groups, channels_per_group))(input_layer)
    
    # Step 2: Permute to shuffle channels
    # Swap the third and fourth dimensions
    permuted = Permute((1, 2, 4, 3))(reshaped)
    
    # Step 3: Reshape back to the original input shape
    reshaped_back = Reshape((32, 32, 3))(permuted)
    
    # Step 4: Flatten and pass through a fully connected layer
    flattened = Flatten()(reshaped_back)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model