import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Add, DepthwiseConv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Depthwise Separable Convolution Layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), padding='same', depth_multiplier=1)(input_layer)
    depthwise_conv = BatchNormalization()(depthwise_conv)
    depthwise_conv = keras.activations.relu(depthwise_conv)
    
    # Flatten the output of depthwise convolution
    flatten_layer = Flatten()(depthwise_conv)
    
    # First fully connected layer
    fc1 = Dense(units=32, activation='relu')(flatten_layer)
    
    # Second fully connected layer
    fc2 = Dense(units=32, activation='relu')(fc1)
    
    # Add the original input with the processed features
    add = Add()([input_layer, fc2])
    
    # Final fully connected layers for classification
    flatten_add = Flatten()(add)
    output_layer = Dense(units=10, activation='softmax')(flatten_add)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])