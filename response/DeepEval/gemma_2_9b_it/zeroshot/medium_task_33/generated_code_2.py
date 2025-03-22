import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # Define input layer
    input_tensor = Input(shape=(32, 32, 3))

    # Split the input into three channels
    def split_channels(inputs):
        r, g, b = tf.split(inputs, 3, axis=-1)
        return r, g, b
    
    r, g, b = Lambda(split_channels, output_shape=(32, 32, 1))(input_tensor)

    # Feature extraction for each channel group
    r_conv = Conv2D(32, (1, 1), activation='relu')(r)
    r_conv = Conv2D(64, (3, 3), activation='relu')(r_conv)
    r_conv = MaxPooling2D((2, 2))(r_conv)

    g_conv = Conv2D(32, (1, 1), activation='relu')(g)
    g_conv = Conv2D(64, (3, 3), activation='relu')(g_conv)
    g_conv = MaxPooling2D((2, 2))(g_conv)

    b_conv = Conv2D(32, (1, 1), activation='relu')(b)
    b_conv = Conv2D(64, (5, 5), activation='relu')(b_conv)
    b_conv = MaxPooling2D((2, 2))(b_conv)

    # Concatenate outputs
    merged = tf.concat([r_conv, g_conv, b_conv], axis=-1)

    # Flatten and fully connected layers
    flattened = Flatten()(merged)
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(10, activation='softmax')(dense2)

    # Create model
    model = Model(inputs=input_tensor, outputs=output)

    return model

# Get the model
model = dl_model()

# Print model summary
model.summary()