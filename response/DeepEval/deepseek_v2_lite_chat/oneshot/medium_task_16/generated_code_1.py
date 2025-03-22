import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
from keras.utils import to_categorical

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along channel dimension
    split1 = Lambda(lambda x: keras.layers.lambda_func.lambda_true(x, lambda x: keras.backend.split(x, 3, axis=1)))(input_layer)
    split2 = Lambda(lambda x: keras.layers.lambda_func.lambda_true(x, lambda x: keras.backend.split(x, 3, axis=1)))(split1)
    split3 = Lambda(lambda x: keras.layers.lambda_func.lambda_true(x, lambda x: keras.backend.split(x, 3, axis=1)))(split2)

    # Apply 1x1 convolutions to each group and average pool
    conv1 = Conv2D(filters=x_train.shape[3] // 3, kernel_size=(1, 1), padding='same')(split1[0])
    conv2 = Conv2D(filters=x_train.shape[3] // 3, kernel_size=(1, 1), padding='same')(split2[0])
    conv3 = Conv2D(filters=x_train.shape[3] // 3, kernel_size=(1, 1), padding='same')(split3[0])
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

    # Concatenate the feature maps
    concat = Concatenate(axis=1)([pool1, pool2, pool3])

    # Flatten and pass through fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1, validation_data=(x_test, y_test))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', test_acc)