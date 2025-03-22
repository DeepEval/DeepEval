import keras
from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPool2D, Conv2DTranspose, Add, Flatten, Dense, AveragePooling2D
from keras.layers.normalization import BatchNormalization

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # 1x1 Convolution Layer
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(inputs)
    
    # 3x3 Depthwise Convolution Layer
    dw_conv1 = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(inputs)
    
    # 1x1 Convolution Layer
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(dw_conv1)
    
    # Stride 2 Average Pooling Layer
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    
    # Fully Connected Layer
    flatten = Flatten()(avg_pool1)
    dense = Dense(units=128, activation='relu')(flatten)
    
    # Output Layer
    outputs = Dense(units=10, activation='softmax')(dense)
    
    # Model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()