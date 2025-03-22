import keras
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Dense, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 1: Spatial feature extraction with depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 2: Layer normalization for stability
    norm = LayerNormalization()(depthwise_conv)
    
    # Step 3: Fully connected layers for channel-wise feature transformation
    flatten = Flatten()(norm)
    fc1 = Dense(units=32 * 32 * 3, activation='relu')(flatten)  # Same number of channels as input
    fc2 = Dense(units=32 * 32 * 3, activation='relu')(fc1)
    
    # Reshape back to image dimensions for addition
    reshaped_fc2 = keras.layers.Reshape((32, 32, 3))(fc2)
    
    # Step 4: Combine original input with processed features
    combined = Add()([input_layer, reshaped_fc2])
    
    # Step 5: Final classification layers
    flatten_combined = Flatten()(combined)
    fc3 = Dense(units=128, activation='relu')(flatten_combined)
    output_layer = Dense(units=10, activation='softmax')(fc3)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model