import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate, BatchNormalization
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # First path: 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second path: sequence of convolutions (1x1 -> 1x7 -> 7x1)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Concatenate the outputs of the two paths
    concat_output = Concatenate()([path1, path2])

    # 1x1 convolution to align the output dimensions with the input image's channel
    main_path_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_output)

    # Branch connecting directly to the input
    branch_output = input_layer

    # Merge the main path output and the branch output through addition
    merged_output = Add()([main_path_output, branch_output])
    
    # Batch normalization
    batch_norm_output = BatchNormalization()(merged_output)
    
    # Flatten the output
    flatten_layer = Flatten()(batch_norm_output)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs=input_layer, outputs=dense2)

    return model