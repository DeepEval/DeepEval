import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Lambda, Dense, Reshape
from keras.models import Model
from keras.regularizers import l2

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocess data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(1, 1), padding='valid')(conv1)
    pool1_flat = Flatten()(pool1)

    # Dropout to mitigate overfitting
    pool1_dropout = BatchNormalization()(pool1_flat)
    pool1_dropout = keras.layers.Dropout(0.5)(pool1_dropout)

    # Second block
    split1 = Lambda(lambda x: keras.layers.split(x, num_or_size_splits=[16, 16, 16, 16])(input_layer) if input_layer.shape.rank == 4 else input_layer)(input_layer)
    pool2 = MaxPooling2D(pool_size=(1, 1), padding='valid')(split1[0])
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='valid')(split1[1])
    pool4 = MaxPooling2D(pool_size=(4, 4), padding='valid')(split1[2])
    pool5 = MaxPooling2D(pool_size=(8, 8), padding='valid')(split1[3])

    # Flatten and connect to dense layers
    concat = Concatenate()([pool2, pool3, pool4, pool5])
    dense1 = Dense(units=512, activation='relu', kernel_regularizer=l2(0.0005))(concat)

    # Reshape for final dense layer
    reshape = Reshape((-1, 512))(dense1)
    dense2 = Dense(units=256, activation='relu')(reshape)
    dense3 = Dense(units=128, activation='relu')(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)