from tensorflow.keras.layers import Input, DepthwiseConv2D, LayerNormalization, Dense, Add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset (for reference, but not used in model construction)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = 10
    
    # Define the input shape for CIFAR-10 images
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Input layer
    inputs = Input(shape=input_shape)

    # Depthwise separable convolutional layer with 7x7 filter
    x = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(inputs)
    
    # Layer normalization
    x = LayerNormalization()(x)

    # Flatten the spatial features for the fully connected layers
    x = Flatten()(x)
    
    # Fully connected layer 1
    fc1 = Dense(units=input_shape[2], activation='relu')(x)  # Channel-wise transformation

    # Fully connected layer 2
    fc2 = Dense(units=input_shape[2], activation='relu')(fc1)  # Channel-wise transformation

    # Flatten the original input to match the dimensions for addition
    flat_inputs = Flatten()(inputs)

    # Adding the original input to the processed features
    added_features = Add()([flat_inputs, fc2])

    # Final fully connected layer for classification
    output = Dense(units=num_classes, activation='softmax')(added_features)

    # Create the model
    model = Model(inputs=inputs, outputs=output)

    # Return the constructed model
    return model

# Example usage
model = dl_model()
model.summary()