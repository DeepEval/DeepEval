import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Add, Concatenate, BatchNormalization, MaxPooling2D, Flatten, Dense
from keras.layers import Permute, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(x):
        # Path 1: 1x1 convolution
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
        # Path 2: 3x3 depthwise separable convolution
        dw_conv = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(x)
        # Path 3: 1x1 convolution
        conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
        return Add()([conv1x1, dw_conv, conv1x1_2])
    
    # Block 2
    def block2(x):
        # Get the shape of the input
        shape = keras.backend.int_shape(x)
        # Reshape to four groups
        reshaped = keras.backend.reshape(x, (-1, shape[-1], shape[1], shape[2], 4))
        # Swap third and fourth dimensions
        permuted = Permute((3, 2, 1, 4))(reshaped)
        # Reshape back to original shape
        x_reshaped = keras.backend.reshape(permuted, shape)
        # Fully connected layer
        dense = Dense(units=128, activation='relu')(x_reshaped)
        return dense
    
    # Construct the model
    input_layer = block1(input_layer)
    output = block2(input_layer)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Optional: Display model summary
model.summary()