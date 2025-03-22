import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.optimizers import Adam


def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Split the CIFAR-10 dataset into three groups along the channel axis
    input_shape = (32, 32, 3)  # Adjust based on your actual dataset's shape
    input_tensor = Input(shape=input_shape)
    split_tensor = Lambda(tf.split)(input_tensor, 3, axis=-1)

    # Feature extraction through convolutional layers
    conv1 = Conv2D(64, (1, 1), activation='relu')(split_tensor[0])
    conv2 = Conv2D(64, (3, 3), activation='relu')(split_tensor[1])
    conv3 = Conv2D(64, (5, 5), activation='relu')(split_tensor[2])

    # Apply dropout to reduce overfitting
    dropout1 = tf.keras.layers.Dropout(0.2)(conv1)
    dropout2 = tf.keras.layers.Dropout(0.2)(conv2)
    dropout3 = tf.keras.layers.Dropout(0.2)(conv3)

    # Concatenate the outputs from the three groups
    concatenate_tensor = concatenate([dropout1, dropout2, dropout3])

    # Second block with four branches
    branch1 = Conv2D(64, (1, 1), activation='relu')(split_tensor[0])
    branch2 = Conv2D(64, (1, 1), activation='relu')(split_tensor[1])
    branch3 = Conv2D(64, (3, 3), activation='relu')(split_tensor[2])
    branch4 = MaxPooling2D(pool_size=(3, 3))(split_tensor[1])

    # Flatten and fully connected layers
    flatten = Flatten()(branch4)
    dense1 = Dense(128, activation='relu')(flatten)
    output = Dense(10, activation='softmax')(dense1)

    # Build the model
    model = Model(inputs=input_tensor, outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model

# Call the function to get the model
model = dl_model()
model.summary()