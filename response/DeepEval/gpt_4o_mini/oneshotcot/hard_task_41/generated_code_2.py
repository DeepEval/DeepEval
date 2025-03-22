import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dropout, Dense, Reshape, Conv2D, Concatenate, BatchNormalization
from keras.models import Model

def dl_model():
    
    # Step 1: Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Block 1
    # Step 2.1: Create three parallel paths with average pooling layers
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    
    # Step 2.2: Flatten and apply dropout to each path
    flattened_path1 = Flatten()(path1)
    dropout_path1 = Dropout(0.5)(flattened_path1)
    
    flattened_path2 = Flatten()(path2)
    dropout_path2 = Dropout(0.5)(flattened_path2)
    
    flattened_path3 = Flatten()(path3)
    dropout_path3 = Dropout(0.5)(flattened_path3)
    
    # Step 2.3: Concatenate the outputs of the three paths
    block1_output = Concatenate()([dropout_path1, dropout_path2, dropout_path3])
    
    # Step 3: Fully connected layer and reshape
    fc_layer = Dense(128, activation='relu')(block1_output)
    reshaped_output = Reshape((4, 4, 8))(fc_layer)  # Reshape to (4, 4, 8) for Block 2

    # Step 4: Block 2
    # Step 4.1: Define the four branches for feature extraction
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

    branch3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch3)

    branch4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(reshaped_output)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(branch4)

    # Step 4.2: Concatenate the outputs of the four branches
    block2_output = Concatenate()([branch1, branch2, branch3, branch4])

    # Step 5: Flatten the output of Block 2
    flatten_block2_output = Flatten()(block2_output)

    # Step 6: Fully connected layers for final classification
    dense1 = Dense(units=128, activation='relu')(flatten_block2_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model