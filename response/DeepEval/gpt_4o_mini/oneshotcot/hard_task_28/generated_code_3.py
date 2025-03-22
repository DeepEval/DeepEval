import keras
from keras.layers import Input, Conv2D, LayerNormalization, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Input layer for CIFAR-10 images (32x32 RGB images)
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Main path - 7x7 depthwise convolution
    main_conv = Conv2D(filters=3, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', use_bias=False)(input_layer)
    
    # Step 3: Layer normalization
    norm_layer = LayerNormalization()(main_conv)
    
    # Step 4: Two sequential 1x1 pointwise convolution layers
    pointwise_conv1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(norm_layer)
    pointwise_conv2 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pointwise_conv1)

    # Step 5: Branch path - connects directly to input
    branch_path = input_layer
    
    # Step 6: Combine main path and branch path with an addition operation
    combined_output = Add()([pointwise_conv2, branch_path])
    
    # Step 7: Flatten the combined output
    flatten_layer = Flatten()(combined_output)
    
    # Step 8: Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Step 9: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model