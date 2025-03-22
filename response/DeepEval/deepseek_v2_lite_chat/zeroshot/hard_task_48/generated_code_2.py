import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Lambda
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model function
def dl_model():
    # Block 1: Feature Extraction
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Convolutional layers for each group
    conv1_1 = split1[0]
    conv1_2 = split1[1]
    conv1_3 = split1[2]
    
    conv1_1 = Conv2D(32, (1, 1), activation='relu', padding='same')(conv1_1)
    conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_2)
    conv1_3 = Conv2D(32, (5, 5), activation='relu', padding='same')(conv1_3)
    
    bn1_1 = BatchNormalization()(conv1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    bn1_3 = BatchNormalization()(conv1_3)
    
    concat1 = concatenate([bn1_1, bn1_2, bn1_3])
    
    # Block 2: Feature Transformation
    branch1 = Conv2D(64, (1, 1), activation='relu', padding='same')(concat1)
    branch2 = MaxPooling2D(pool_size=(3, 3))(concat1)
    
    branch3 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch1)
    branch4 = Lambda(lambda x: tf.keras.layers.average(x))(branch1)
    branch4 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch4)
    
    branch5 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch2)
    branch5 = Lambda(lambda x: tf.keras.layers.average(x))(branch5)
    branch5 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch5)
    
    branch6 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch3)
    branch6 = Lambda(lambda x: tf.keras.layers.average(x))(branch6)
    branch6 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch6)
    
    branch7 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch3)
    branch7 = Lambda(lambda x: tf.keras.layers.average(x))(branch7)
    branch7 = Conv2D(64, (3, 3), activation='relu')(branch7)
    branch7 = Lambda(lambda x: tf.keras.layers.average(x))(branch7)
    branch7 = Lambda(lambda x: tf.keras.layers.average(x))(branch7)
    
    branch8 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch3)
    branch8 = Lambda(lambda x: tf.keras.layers.average(x))(branch8)
    branch8 = Conv2D(64, (3, 3), activation='relu')(branch8)
    branch8 = Lambda(lambda x: tf.keras.layers.average(x))(branch8)
    branch8 = Lambda(lambda x: tf.keras.layers.average(x))(branch8)
    
    concat2 = concatenate([branch1, branch2, branch3, branch4, branch5, branch6, branch7, branch8])
    
    # Classification head
    flatten = Flatten()(concat2)
    output = Dense(10, activation='softmax')(flatten)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and return the model
model = dl_model()