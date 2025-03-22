import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Model architecture
def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the last dimension
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    split2 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    split3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    
    # Depthwise separable convolutions
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split1[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(split2[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(split3[2])
    
    # Concatenate outputs from different groups
    concat = Concatenate(axis=-1)([conv1, conv2, conv3])
    
    # Feature extraction branches
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(concat)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(concat)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(concat)
    branch4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(concat)
    
    # Concatenate outputs from branches
    fuse = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    
    # Flatten and fully connected layers
    flatten = Flatten()(fuse)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', test_acc)