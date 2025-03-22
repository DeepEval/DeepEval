import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    def block1(x):
        # 3x3 convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        # Two 1x1 convolutional layers
        conv2 = Conv2D(filters=64, kernel_size=(1, 1))(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(1, 1))(conv2)
        # Average pooling
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv3)
        # Dropout layer to mitigate overfitting
        drop1 = Dropout(0.5)(pool1)
        return drop1
    
    # Second block
    def block2(x):
        # Average pooling
        pool2 = AveragePooling2D(pool_size=(2, 2))(x)
        # Dropout layer to mitigate overfitting
        drop2 = Dropout(0.5)(pool2)
        # Global average pooling
        avg_pool = AveragePooling2D(pool_size=(7, 7))(drop2)
        return avg_pool
    
    # Combine the outputs of the blocks
    x = block1(input_layer)
    x = block2(x)
    
    # Flatten and fully connected layers
    flatten = Flatten()(x)
    dense = Dense(units=128, activation='relu')(flatten)
    output = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate and return the constructed model
model = dl_model()