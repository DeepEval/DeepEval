from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Model architecture
def dl_model():
    # Main path
    main_input = Input(shape=(32, 32, 3))
    x = Conv2D(64, kernel_size=3, activation='relu')(main_input)
    x = Conv2D(64, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    
    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    branch_x = Conv2D(128, kernel_size=5, activation='relu')(branch_input)
    
    # Combine features from both paths
    concat = Concatenate()([x, branch_x])
    concat = BatchNormalization()(concat)
    
    # Fully connected layers
    x = Flatten()(concat)
    x = Dense(512, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    
    # Model
    model = Model(inputs=[main_input, branch_input], outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Build the model
model = dl_model()

# Summary of the model
model.summary()