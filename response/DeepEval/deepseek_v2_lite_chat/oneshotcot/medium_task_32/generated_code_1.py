import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
import tensorflow as tf

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32

    # Split the input into three groups
    inputs = Input(shape=input_shape)
    a, b, c = tf.split(inputs, 3, axis=-1)
    
    # Feature extraction for each group
    def depthwise_separable_conv(input_tensor, kernel_size):
        # Depthwise convolution
        dw = Conv2D(filters=kernel_size**2, kernel_size=kernel_size, padding='same', use_bias=False)(input_tensor)
        # Pointwise convolution
        dp = Conv2D(filters=3, kernel_size=1, padding='same', use_bias=False)(dw)
        return dp
    
    # Use depthwise separable conv layers with different kernel sizes for each group
    a_conv = depthwise_separable_conv(a, 1)
    b_conv = depthwise_separable_conv(b, 3)
    c_conv = depthwise_separable_conv(c, 5)
    
    # Concatenate the outputs from each group
    concat = Concatenate(axis=-1)([a_conv, b_conv, c_conv])
    
    # Flatten and pass through fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())