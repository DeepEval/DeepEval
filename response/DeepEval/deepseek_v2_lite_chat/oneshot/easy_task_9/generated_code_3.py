import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 Convolution to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Depthwise separable convolution for feature extraction
    depthwise = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(conv1)
    depthwise_sep = keras.layers.BatchNormalization()(depthwise)
    depthwise_sep = keras.layers.Activation('relu')(depthwise_sep)
    
    # 1x1 Convolution to reduce dimensionality
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(depthwise_sep)
    
    # Add the output of the 1x1 convolution to the original input layer
    add = Add()([input_layer, conv2])
    
    # MaxPooling layer
    pool = MaxPooling2D(pool_size=(2, 2))(add)
    
    # Flatten the output
    flatten = Flatten()(pool)
    
    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flatten)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Print model summary
model.summary()