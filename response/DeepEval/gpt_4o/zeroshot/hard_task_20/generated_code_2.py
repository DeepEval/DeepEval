import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Add, Lambda, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)
    num_classes = 10
    
    # Input layer
    inputs = Input(shape=input_shape)

    # Main path with splitting and convolutional layers
    # Split the input into three parts along the channel dimension
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Convolutional layers with different kernel sizes
    conv1 = Conv2D(32, (1, 1), padding='same', activation='relu')(split_inputs[0])
    conv3 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_inputs[1])
    conv5 = Conv2D(32, (5, 5), padding='same', activation='relu')(split_inputs[2])

    # Concatenate the outputs of the convolutions
    main_path_output = Concatenate()([conv1, conv3, conv5])

    # Branch path with a 1x1 convolutional layer
    branch_output = Conv2D(96, (1, 1), padding='same', activation='relu')(inputs)

    # Fuse the features from the main and branch paths
    fused_features = Add()([main_path_output, branch_output])

    # Flatten the fused features
    flattened = Flatten()(fused_features)

    # Classification with fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    outputs = Dense(num_classes, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create and summarize the model
model = dl_model()
model.summary()

# Optionally, train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))