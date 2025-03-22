import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from tensorflow.keras.regularizers import l2

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Reshape the data
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Function to build the deep learning model
def dl_model():
    # Input layer
    input_A = Input(shape=(28, 28, 1))
    input_B = Input(shape=(28, 28, 1))

    # Two 1x1 convolutional layers
    conv1 = Conv2D(64, (1, 1), activation='relu', kernel_regularizer=l2(0.01))(input_A)
    conv2 = Conv2D(64, (1, 1), activation='relu', kernel_regularizer=l2(0.01))(input_B)

    # 3x1 convolutional layer
    conv3 = Conv2D(64, (3, 1), padding='same', activation='relu')(conv1)

    # 1x3 convolutional layer
    conv4 = Conv2D(64, (1, 3), padding='same', activation='relu')(conv2)

    # Restore channels
    conv3_expand = Conv2D(64, (1, 1), activation='relu')(conv3)
    conv4_expand = Conv2D(64, (1, 1), activation='relu')(conv4)

    # Combine features via addition
    combined = Concatenate()([conv3_expand, conv4_expand])

    # Dropout layer for regularization
    dropout = tf.keras.layers.Dropout(0.5)(combined)

    # Flatten layer
    flattened = Flatten()(dropout)

    # Fully connected layer
    fc1 = Dense(128, activation='relu')(flattened)

    # Output layer
    output = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=[input_A, input_B], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()
print(model.summary())