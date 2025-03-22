import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have dimensions of 32x32 with 3 channels

    # Main pathway with 1x1 convolution
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Parallel branch
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate the outputs from the parallel branches
    parallel_output = Concatenate()([path1, path2, path3])
    
    # Combine the main pathway and the parallel branch
    combined_output = Add()([main_path, parallel_output])

    # Apply a 1x1 convolution to the combined output
    final_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(combined_output)

    # Batch normalization
    batch_norm = BatchNormalization()(final_conv)

    # Flatten the output
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=dense2)

    return model

# Create the model
model = dl_model()
model.summary()  # Optional: print the model summary to see the architecture