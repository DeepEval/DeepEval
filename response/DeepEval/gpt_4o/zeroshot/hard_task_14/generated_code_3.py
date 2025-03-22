import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    # Global Average Pooling
    x_main = GlobalAveragePooling2D()(input_layer)
    
    # Two fully connected layers
    x_main = Dense(units=64, activation='relu')(x_main)
    x_main = Dense(units=3, activation='relu')(x_main)
    
    # Reshape to match input layer shape
    x_main = Dense(units=32 * 32 * 3, activation='sigmoid')(x_main)
    x_main = keras.layers.Reshape((32, 32, 3))(x_main)
    
    # Element-wise multiplication with the original feature map
    x_main = Multiply()([input_layer, x_main])

    # Branch path
    # 3x3 Convolution to adjust the channels
    x_branch = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Combine the main and branch paths
    x = Add()([x_main, x_branch])

    # Fully connected layers to produce final output
    x = Flatten()(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage: Load CIFAR-10 data and compile the model
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = dl_model()
model.summary()