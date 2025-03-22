import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Conv2DTranspose
from keras.models import Model

def dl_model():
    # Define the main path
    input_layer = Input(shape=(32, 32, 3))
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Define the branch path
    branch_conv1x1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # First branch: local features with 3x3 convolution
    branch_local = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_conv1x1)
    
    # Second branch: downsample with average pooling, then process with 3x3 convolution, and upsample with transpose convolution
    branch_downsample = AveragePooling2D(pool_size=(2, 2), strides=2)(branch_conv1x1)
    branch_upsample = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch_downsample)
    
    # Third branch: similar to the second branch
    branch_downsample2 = AveragePooling2D(pool_size=(2, 2), strides=2)(branch_conv1x1)
    branch_upsample2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch_downsample2)
    
    # Concatenate outputs of all branches
    concatenated = Concatenate()([branch_local, branch_upsample, branch_upsample2])
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    # Batch normalization and flatten
    batch_norm = BatchNormalization()(main_path_output)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()