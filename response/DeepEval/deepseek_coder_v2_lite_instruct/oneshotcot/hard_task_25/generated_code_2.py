import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add, Conv2DTranspose
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Branch path
    branch_path_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # First branch (local features with 3x3 convolution)
    branch_local = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch_path_conv)
    
    # Second branch (downsample with average pooling, then process with 3x3 convolution, and upsample with transpose convolutional layer)
    branch_downsampled = AveragePooling2D(pool_size=(2, 2))(branch_path_conv)
    branch_downsampled = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch_downsampled)
    branch_upsampled = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch_downsampled)
    
    # Third branch (similar to the second branch but uses max pooling instead of average pooling for downsampling)
    branch_downsampled_max = MaxPooling2D(pool_size=(2, 2))(branch_path_conv)
    branch_downsampled_max = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch_downsampled_max)
    branch_upsampled_max = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch_downsampled_max)
    
    # Concatenate outputs of all branches
    concatenated = Concatenate()([branch_local, branch_upsampled, branch_upsampled_max])
    
    # Apply a 1x1 convolutional layer to the concatenated output
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concatenated)
    
    # Add batch normalization
    batch_norm = BatchNormalization()(main_path_output)
    
    # Flatten the output
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model