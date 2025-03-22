import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel dimension
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    splitted = Lambda(split_channels)(input_layer)
    
    # Define a function to apply the series of convolutions
    def conv_series(x):
        x = Conv2D(32, (1, 1), activation='relu')(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (1, 1), activation='relu')(x)
        return x
    
    # Apply the convolutions to each group
    conv_outputs = [conv_series(group) for group in splitted]
    
    # Combine the outputs from the three groups using addition
    combined = Add()(conv_outputs)
    
    # Fuse the combined output with the original input
    fused = Add()([combined, input_layer])
    
    # Flatten the result and pass it through a fully connected layer
    flat = Flatten()(fused)
    output_layer = Dense(10, activation='softmax')(flat)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model