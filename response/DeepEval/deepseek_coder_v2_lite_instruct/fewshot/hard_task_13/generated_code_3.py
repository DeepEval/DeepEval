import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block_1(input_tensor):
        # Parallel branches
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool3x3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)

        # Concatenate the outputs of the branches
        output_tensor = Concatenate()([conv1x1, conv3x3, conv5x5, pool3x3])
        return output_tensor

    def block_2(input_tensor):
        # Global average pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        
        # Fully connected layers
        fc1 = Dense(units=64, activation='relu')(gap)
        fc2 = Dense(units=10, activation='softmax')(fc1)  # Output layer
        
        # Reshape weights to match the input shape
        weights = Reshape((32, 32, 3))(fc2)
        
        # Element-wise multiplication with the input feature map
        output_tensor = Multiply()([input_tensor, weights])
        return output_tensor

    # Apply the first block
    block1_output = block_1(input_tensor=input_layer)
    
    # Apply the second block
    model_output = block_2(input_tensor=block1_output)

    # Create the model
    model = Model(inputs=input_layer, outputs=model_output)

    return model

# Create the model
model = dl_model()
model.summary()