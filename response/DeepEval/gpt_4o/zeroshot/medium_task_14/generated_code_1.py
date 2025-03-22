import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Sequential blocks with separate outputs
    conv1 = Conv2D(32, (3, 3), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)
    
    conv2 = Conv2D(64, (3, 3), padding='same')(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = ReLU()(bn2)
    
    conv3 = Conv2D(128, (3, 3), padding='same')(relu2)
    bn3 = BatchNormalization()(conv3)
    relu3 = ReLU()(bn3)

    # Parallel branch to process input directly
    parallel_conv1 = Conv2D(32, (1, 1), padding='same')(input_layer)
    parallel_bn1 = BatchNormalization()(parallel_conv1)
    parallel_relu1 = ReLU()(parallel_bn1)

    parallel_conv2 = Conv2D(64, (3, 3), padding='same')(parallel_relu1)
    parallel_bn2 = BatchNormalization()(parallel_conv2)
    parallel_relu2 = ReLU()(parallel_bn2)

    parallel_conv3 = Conv2D(128, (3, 3), padding='same')(parallel_relu2)
    parallel_bn3 = BatchNormalization()(parallel_conv3)
    parallel_relu3 = ReLU()(parallel_bn3)

    # Adding outputs from sequential blocks and parallel branch
    merged = Add()([relu3, parallel_relu3])

    # Fully connected layers for classification
    flatten = Flatten()(merged)
    fc1 = Dense(256, activation='relu')(flatten)
    fc2 = Dense(128, activation='relu')(fc1)
    
    # Output layer for 10 classes (CIFAR-10)
    output_layer = Dense(10, activation='softmax')(fc2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Initialize and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()