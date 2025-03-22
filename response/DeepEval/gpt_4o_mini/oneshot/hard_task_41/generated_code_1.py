import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, Conv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Three parallel paths with different average pooling sizes
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten and apply dropout to each path
    flattened_path1 = Flatten()(path1)
    dropout_path1 = Dropout(0.5)(flattened_path1)

    flattened_path2 = Flatten()(path2)
    dropout_path2 = Dropout(0.5)(flattened_path2)

    flattened_path3 = Flatten()(path3)
    dropout_path3 = Dropout(0.5)(flattened_path3)

    # Concatenate the outputs of the three paths
    concatenated_output = Concatenate()([dropout_path1, dropout_path2, dropout_path3])

    # Fully connected layer to transform into 4D tensor
    dense_layer1 = Dense(units=128, activation='relu')(concatenated_output)
    reshaped_output = Reshape((1, 1, 128))(dense_layer1)

    # Block 2: Feature extraction with multiple branches
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    
    branch5 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(reshaped_output)
    branch6 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch5)

    # Concatenate the outputs of the branches
    block2_output = Concatenate()([branch1, branch4, branch6])

    # Final flattening and fully connected layers for classification
    flatten_block2 = Flatten()(block2_output)
    dense_layer2 = Dense(units=64, activation='relu')(flatten_block2)
    output_layer = Dense(units=10, activation='softmax')(dense_layer2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model