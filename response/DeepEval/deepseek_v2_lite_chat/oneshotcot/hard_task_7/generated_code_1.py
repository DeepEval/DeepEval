import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolutional layer with 32 kernels
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')(input_layer)
    
    # Splitting the input into two groups
    split = Lambda(lambda x: keras.backend.split(x, 2, axis=-1))(conv1)
    
    # Operations for Block 1
    def block1():
        group1 = split[0]
        group2 = split[1]
        
        # 1x1 convolution
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        # Depthwise separable convolution
        conv1_2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same')(group1)
        conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_2)
        # 1x1 convolution
        conv1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(conv1_2)
        
        # Merge paths
        merged = Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])
        
        return merged
    
    # Operations for Block 2
    def block2():
        # Reshape and shuffle channels
        input_shape = keras.backend.int_shape(block1())
        batch_size = keras.backend.shape_at(input_shape, 0)
        height, width, channels = keras.backend.int_shape(block1()) // batch_size
        channels_per_group = channels // 4
        reshaped = keras.backend.reshape(block1(), (batch_size, height, width, channels_per_group, 4))
        reshaped = keras.backend.permute_dimensions(reshaped, (0, 3, 1, 2, 4))
        reshaped = keras.backend.reshape(reshaped, (batch_size, height * width * channels_per_group))
        
        # Fully connected layer
        dense = Dense(units=128, activation='relu')(reshaped)
        
        return dense
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=block2())
    
    return model

# Construct the model
model = dl_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])