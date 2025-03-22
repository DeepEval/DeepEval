from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.applications.vgg16 import VGG16
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape data to include channel dimension
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    
    # Normalize data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Convert labels to categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # Feature extraction using VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(28, 28, 1))
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers
    input_layer = Input(shape=(28, 28, 1))
    x = Conv2D(64, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten and feed into fully connected layers
    x = Flatten()(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    output = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model