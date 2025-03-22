import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Permute

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image shape
    input_layer = Input(shape=input_shape)
    
    # Reshape and shuffle channels to enable convolutions on different groups
    permuted_layer = Permute((2, 3, 1))(input_layer)  # Swap channels to the second dimension
    reshaped_layer = Permute((2, 3, 1))(permuted_layer)  # Swap back to the original input shape
    
    # Add convolutional layers
    conv1 = Conv2D(32, (3, 3), activation='relu')(reshaped_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Flatten and pass through dense layers
    flat = Flatten()(pool)
    dense1 = Dense(512, activation='relu')(flat)
    output = Dense(10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    return model

# Build the model
model = dl_model()
model.summary()