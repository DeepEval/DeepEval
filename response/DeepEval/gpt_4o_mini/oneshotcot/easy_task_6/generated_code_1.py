import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Main path
    # First convolutional layer to increase feature map width
    main_path_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Second convolutional layer to restore the number of channels
    main_path_conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv1)
    
    # Step 3: Branch path directly connects to input
    branch_path = input_layer
    
    # Step 4: Combine paths using addition operation
    combined_output = Add()([main_path_conv2, branch_path])
    
    # Step 5: Flatten layer
    flatten_layer = Flatten()(combined_output)
    
    # Step 6: Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model