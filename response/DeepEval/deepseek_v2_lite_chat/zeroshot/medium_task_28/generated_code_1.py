import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, LayerNormalization, ReLU, Add, Flatten, Dense, Concatenate
from keras.layers import Convolution2D, MaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Model architecture
def dl_model():
    # First, generate attention weights with a 1x1 convolution
    input_layer = Input(shape=(32, 32, 3))
    attention_weights = Conv2D(1, (1, 1), activation='softmax')(input_layer)
    
    # Multiply attention weights with input features
    weighted_input = Conv2D(1, (1, 1), activation='multiply', name='weighted_input')(input_layer)
    weighted_input = keras.layers.Multiply()([weighted_input, attention_weights])
    
    # Reduce dimensionality with a 1x1 convolution
    reduced_input = Conv2D(int(input_layer.shape[-1]/3), (1, 1), activation='relu')(weighted_input)
    reduced_input = LayerNormalization(epsilon=1e-6)(reduced_input)
    
    # Expand dimensions with another 1x1 convolution
    expanded_input = Conv2D(int(input_layer.shape[-1]), (1, 1), activation='relu')(reduced_input)
    
    # Add expanded input to original input
    output = Add()([input_layer, expanded_input])
    
    # Flatten and classify
    flattened = Flatten()(output)
    output = Dense(10, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)