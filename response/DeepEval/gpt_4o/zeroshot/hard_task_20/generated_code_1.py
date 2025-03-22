import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Concatenate, Add, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize the images
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    # Split into three groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply different convolutions
    conv_1x1 = Conv2D(32, (1, 1), activation='relu', padding='same')(split_layer[0])
    conv_3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(split_layer[1])
    conv_5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(split_layer[2])

    # Concatenate outputs
    concat_layer = Concatenate()([conv_1x1, conv_3x3, conv_5x5])

    # Branch path
    branch_conv = Conv2D(96, (1, 1), activation='relu', padding='same')(input_layer)

    # Combine main and branch paths
    fused_features = Add()([concat_layer, branch_conv])

    # Flatten and add dense layers for classification
    flattened = Flatten()(fused_features)
    dense_1 = Dense(128, activation='relu')(flattened)
    output_layer = Dense(10, activation='softmax')(dense_1)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of using the model
model = dl_model()
model.summary()