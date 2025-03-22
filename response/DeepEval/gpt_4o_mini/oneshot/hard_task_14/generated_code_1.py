import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Main Path
    main_path = GlobalAveragePooling2D()(input_layer)  # Global average pooling
    main_path = Dense(units=512, activation='relu')(main_path)  # First fully connected layer
    main_path = Dense(units=3, activation='sigmoid')(main_path)  # Generate weights with size equal to channels (3)
    weights = Reshape((1, 1, 3))(main_path)  # Reshape weights to match input shape for multiplication
    main_path = Multiply()([input_layer, weights])  # Element-wise multiplication with the original feature map

    # Branch Path
    branch_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)  # Adjust channels with 3x3 convolution
    
    # Combine paths
    combined_output = Add()([main_path, branch_path])  # Add both paths together
    
    # Final classification layers
    combined_output = Flatten()(combined_output)  # Flatten the combined output
    dense1 = Dense(units=128, activation='relu')(combined_output)  # First fully connected layer
    dense2 = Dense(units=64, activation='relu')(dense1)  # Second fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense2)  # Output layer for 10 classes in CIFAR-10

    model = Model(inputs=input_layer, outputs=output_layer)  # Create the Keras model

    return model