import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def dl_model():
    # Main path input
    input_main = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_main)
    
    # Depthwise separable convolutional layers
    x1 = Conv2D(32, (1, 1), padding="same", activation="relu")(x[0])
    x2 = Conv2D(32, (3, 3), padding="same", activation="relu")(x[1])
    x3 = Conv2D(32, (5, 5), padding="same", activation="relu")(x[2])
    
    # Concatenate outputs from the three groups
    x_main = Add()([x1, x2, x3])
    
    # Branch path
    x_branch = Conv2D(64, (1, 1), padding="same", activation="relu")(x_main)
    
    # Align the number of output channels
    x_branch = Conv2D(64, (1, 1), padding="same", activation="relu")(x_branch)
    
    # Fully connected layers
    x_branch = Flatten()(x_branch)
    x_branch = Dense(512, activation="relu")(x_branch)
    x_branch = Dense(10, activation="softmax")(x_branch)
    
    # Combine outputs from the main and branch paths
    output = Add()([x_main, x_branch])
    
    # Model
    model = Model(inputs=input_main, outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model