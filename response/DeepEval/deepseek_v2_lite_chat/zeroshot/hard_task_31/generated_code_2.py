from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Create the input layers for the Functional API
    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Block 1: Main and Branch Path
    def main_path(x):
        # Expand the feature map and restore channels
        x = layers.Conv2D(32, 3, activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.Upping_channels_conv(x)
        x = layers.Conv2D(64, 3, activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2)(x)
        return x
    
    def branch_path(x):
        return x
    
    x = main_path(input_layer)
    x = branch_path(x)
    
    # Block 2: Separable Convolution Layers
    def separate_convs(x):
        split1, split2, split3 = keras.layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
        x = layers.SeparableConv2D(128, 3, activation='relu')(split1)
        x = layers.SeparableConv2D(128, 3, activation='relu')(split2)
        x = layers.SeparableConv2D(128, 3, activation='relu')(split3)
        x = layers.Concatenate()([x, split1, split2, split3])
        x = layers.Dropout(0.2)(x)
        return x
    
    x = separate_convs(x)
    
    # Output Layer
    x = layers.Flatten()(x)
    output = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model = keras.Model(input_layer, output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model