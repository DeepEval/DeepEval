import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Flatten, Concatenate, Dense
from keras.datasets import cifar10
from keras.models import Model

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
input_layer = Input(shape=(32, 32, 3))

# First block
def block1():
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
    return pool1

# Second block
def block2():
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(block1().output)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
    return pool2

# Third block
def block3():
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(block2().output)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)
    return pool3

# Direct branch
branch_D = Conv2D(64, (1, 1), activation='relu')(input_layer)

# Concatenate outputs from blocks and branch_D
concat = Concatenate()([block3().output, branch_D])

# Fully connected layers
dense1 = Dense(units=512, activation='relu')(concat)
dense2 = Dense(units=256, activation='relu')(dense1)
output_layer = Dense(units=10, activation='softmax')(dense2)

# Model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

return model