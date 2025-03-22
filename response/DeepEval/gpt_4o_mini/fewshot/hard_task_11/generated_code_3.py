import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Main pathway
    main_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch path
    branch_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch_conv2 = Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(input_layer)
    branch_conv3 = Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(input_layer)
    
    # Concatenate outputs from the branch
    branch_output = Concatenate()([branch_conv1, branch_conv2, branch_conv3])
    
    # Concatenate the main path output and the branch output
    concatenated = Concatenate()([main_conv1, branch_output])
    
    # Final convolution to adjust the number of channels
    final_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Add the input to the final output (skip connection)
    added_output = Add()([input_layer, final_conv])

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # This will print the model architecture