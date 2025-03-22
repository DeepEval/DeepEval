from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Define input
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    # Branch pathway
    y = Conv2D(32, (2, 2), activation='relu', padding='same')(input_layer)

    # Fuse both pathways
    z = Add()([x, y])

    # Global average pooling and classification
    z = GlobalAveragePooling2D()(z)
    z = Flatten()(z)
    output_layer = Dense(10, activation='softmax')(z)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Instantiate and train the model
model = dl_model()
model.summary()
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)