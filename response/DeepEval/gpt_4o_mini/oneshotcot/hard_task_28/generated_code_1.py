import keras
from keras.layers import Input, Conv2D, LayerNormalization, Add, Flatten, Dense

def dl_model():
    # Step 1: Input Layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Step 2: Main Path
    # Step 2.1: 7x7 Depthwise Convolution
    depthwise_conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', groups=3)(input_layer)
    
    # Step 2.2: Layer Normalization
    norm_layer = LayerNormalization()(depthwise_conv)

    # Step 2.3: Two Sequential 1x1 Pointwise Convolution Layers
    pointwise_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(norm_layer)
    pointwise_conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pointwise_conv1)

    # Step 3: Branch Path
    # This path connects directly to the input
    branch_path = input_layer

    # Step 4: Combine Outputs of Both Paths
    combined_output = Add()([pointwise_conv2, branch_path])

    # Step 5: Flatten the Combined Output
    flatten_layer = Flatten()(combined_output)

    # Step 6: Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Step 7: Build the Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model