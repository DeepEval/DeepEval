import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, ReLU, Add, Flatten, Dense

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Preprocess the data
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Define the main path
    input_main = Input(shape=(28, 28, 1))
    
    x = Conv2D(32, (3, 3), padding='same')(input_main)
    x = ReLU()(x)
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Define the branch path
    input_branch = Input(shape=(28, 28, 1))
    
    y = Conv2D(16, (1, 1), padding='same')(input_branch)
    y = ReLU()(y)
    
    # Merge the outputs from both paths
    combined = Add()([x, y])
    
    # Flatten the output and add the final layers
    z = Flatten()(combined)
    output = Dense(10, activation='softmax')(z)
    
    # Construct the model
    model = Model(inputs=[input_main, input_branch], outputs=output)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()