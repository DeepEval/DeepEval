import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Set the seed for reproducibility
seed = 42
tf.random.set_seed(seed)


def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split input into three groups
    split1 = Lambda(lambda x: tf.split(x, [1, 1, 1], axis=-1))(inputs)
    
    # Feature extraction with different kernel sizes
    conv1 = Conv2D(64, (1, 1), padding='same')(split1[0])  # 1x1 kernel
    conv2 = Conv2D(64, (3, 3), padding='same')(split1[1])  # 3x3 kernel
    conv3 = Conv2D(64, (5, 5), padding='same')(split1[2])  # 5x5 kernel
    
    # Concatenate the outputs
    concat = tf.concat([conv1, conv2, conv3], axis=-1)
    
    # Branch path
    branch = Conv2D(64, (1, 1), padding='same')(inputs)  # 1x1 kernel to match dimensions
    
    # Combine main path and branch path
    fused = Add()([concat, branch])
    
    # Flatten and pass through two fully connected layers
    x = Flatten()(fused)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Get the constructed model
model = dl_model()