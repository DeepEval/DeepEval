import keras
from keras.layers import Input, Conv2D, LayerNormalization, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)
    
    # Main path
    depthwise_conv = Conv2D(filters=3, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', groups=3)(input_layer)
    layer_norm = LayerNormalization()(depthwise_conv)
    
    pointwise_conv1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(layer_norm)
    pointwise_conv2 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pointwise_conv1)
    
    # Branch path
    branch_path = input_layer  # Direct connection to the input
    
    # Combine main path and branch path
    combined_output = Add()([pointwise_conv2, branch_path])

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model