import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Reshape


def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the output shape
    output_shape = (10,)

    # Define the number of filters and kernel sizes for each convolutional layer
    num_filters = [16, 32, 64, 128]
    kernel_sizes = [3, 5, 7, 9]

    # Define the number of units in the fully connected layers
    num_units = [128, 64]

    # Define the dropout rate
    dropout_rate = 0.5

    # Define the batch normalization parameters
    batch_normalization = True
    batch_size = 128

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the first block
    block_1 = input_layer

    # Define the first parallel path
    path_1 = Conv2D(num_filters[0], kernel_size=kernel_sizes[0], activation='relu')(block_1)
    path_1 = MaxPooling2D(pool_size=2)(path_1)
    path_1 = Flatten()(path_1)
    path_1 = Dropout(dropout_rate)(path_1)

    # Define the second parallel path
    path_2 = Conv2D(num_filters[1], kernel_size=kernel_sizes[1], activation='relu')(block_1)
    path_2 = MaxPooling2D(pool_size=2)(path_2)
    path_2 = Flatten()(path_2)
    path_2 = Dropout(dropout_rate)(path_2)

    # Define the third parallel path
    path_3 = Conv2D(num_filters[2], kernel_size=kernel_sizes[2], activation='relu')(block_1)
    path_3 = MaxPooling2D(pool_size=2)(path_3)
    path_3 = Flatten()(path_3)
    path_3 = Dropout(dropout_rate)(path_3)

    # Define the fourth parallel path
    path_4 = Conv2D(num_filters[3], kernel_size=kernel_sizes[3], activation='relu')(block_1)
    path_4 = MaxPooling2D(pool_size=2)(path_4)
    path_4 = Flatten()(path_4)
    path_4 = Dropout(dropout_rate)(path_4)

    # Concatenate the parallel paths
    block_1 = Concatenate()([path_1, path_2, path_3, path_4])

    # Define the second block
    block_2 = block_1

    # Define the first branch
    branch_1 = Conv2D(num_filters[0], kernel_size=kernel_sizes[0], activation='relu')(block_2)
    branch_1 = MaxPooling2D(pool_size=2)(branch_1)
    branch_1 = Flatten()(branch_1)
    branch_1 = Dropout(dropout_rate)(branch_1)

    # Define the second branch
    branch_2 = Conv2D(num_filters[1], kernel_size=kernel_sizes[1], activation='relu')(block_2)
    branch_2 = MaxPooling2D(pool_size=2)(branch_2)
    branch_2 = Flatten()(branch_2)
    branch_2 = Dropout(dropout_rate)(branch_2)

    # Define the third branch
    branch_3 = Conv2D(num_filters[2], kernel_size=kernel_sizes[2], activation='relu')(block_2)
    branch_3 = MaxPooling2D(pool_size=2)(branch_3)
    branch_3 = Flatten()(branch_3)
    branch_3 = Dropout(dropout_rate)(branch_3)

    # Define the fourth branch
    branch_4 = Conv2D(num_filters[3], kernel_size=kernel_sizes[3], activation='relu')(block_2)
    branch_4 = MaxPooling2D(pool_size=2)(branch_4)
    branch_4 = Flatten()(branch_4)
    branch_4 = Dropout(dropout_rate)(branch_4)

    # Concatenate the branches
    block_2 = Concatenate()([branch_1, branch_2, branch_3, branch_4])

    # Define the fully connected layers
    fc_layer_1 = Dense(num_units[0], activation='relu')(block_2)
    fc_layer_2 = Dense(num_units[1], activation='relu')(fc_layer_1)

    # Define the output layer
    output_layer = Dense(output_shape[0], activation='softmax')(fc_layer_2)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model