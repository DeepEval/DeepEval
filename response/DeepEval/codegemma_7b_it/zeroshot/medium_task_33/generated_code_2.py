from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Define the model
    inputs = Input(shape=(32, 32, 3))
    splited = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    group_1 = Lambda(lambda x: Conv2D(32, (1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0))(x))(splited[0])
    group_2 = Lambda(lambda x: SeparableConv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0))(x))(splited[1])
    group_3 = Lambda(lambda x: SeparableConv2D(32, (5, 5), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0))(x))(splited[2])

    merged = Concatenate()([group_1, group_2, group_3])

    x = AveragePooling2D()(merged)
    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_initializer=glorot_uniform(seed=0))(x)
    x = Dense(64, activation='relu', kernel_initializer=glorot_uniform(seed=0))(x)
    outputs = Dense(10, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

    return model

model = dl_model()