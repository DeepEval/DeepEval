import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, Add, Flatten
from keras.models import Model

def dl_model():

    # Input layer for CIFAR-10 images (32x32x3)
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = GlobalAveragePooling2D()(input_layer)  # Global average pooling
    main_path = Dense(units=512, activation='relu')(main_path)  # Fully connected layer
    main_path = Dense(units=3, activation='sigmoid')(main_path)  # Output layer for channel weights
    weights = Reshape((1, 1, 3))(main_path)  # Reshape to match input layer shape
    main_path_output = Multiply()([input_layer, weights])  # Element-wise multiplication with original feature map

    # Branch path
    branch_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)  # 3x3 conv

    # Merging both paths
    combined_output = Add()([main_path_output, branch_path])  # Element-wise addition

    # Fully connected layers for classification
    flatten_layer = Flatten()(combined_output)  # Flattening the combined output
    dense1 = Dense(units=512, activation='relu')(flatten_layer)  # Fully connected layer
    dense2 = Dense(units=256, activation='relu')(dense1)  # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense2)  # Output layer for 10 classes

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model