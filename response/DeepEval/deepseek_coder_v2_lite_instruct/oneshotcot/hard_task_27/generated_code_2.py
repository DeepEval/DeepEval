import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense, DepthwiseConv2D, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # 7x7 depthwise separable convolutional layer
    conv_depthwise = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(input_layer)
    conv_depthwise = BatchNormalization()(conv_depthwise)
    
    # Fully connected layer for channel-wise feature transformation
    fc1 = Dense(units=32, activation='relu')(GlobalAveragePooling2D()(conv_depthwise))
    
    # Another fully connected layer for channel-wise feature transformation
    fc2 = Dense(units=32, activation='relu')(fc1)
    
    # Addition operation to combine original input with processed features
    added = Add()([input_layer, fc2])
    
    # Flatten layer
    flatten_layer = Flatten()(added)
    
    # Final two fully connected layers for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and return the model
model = dl_model()