import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Conv2DTranspose, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Function to create the model
def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First branch: Local feature extraction
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    # Pooling layer to reduce spatial dimensions
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Second branch: Average pooling and subsequent upsampling
    avg_pool1 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    conv3 = Conv2D(64, (3, 3), activation='relu')(avg_pool1)
    upconv1 = UpSampling2D(size=2)(conv3)
    concat1 = Concatenate()([pool1, upconv1])
    
    # Third branch: Average pooling and subsequent upsampling
    avg_pool2 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    conv4 = Conv2D(64, (3, 3), activation='relu')(avg_pool2)
    upconv2 = UpSampling2D(size=2)(conv4)
    concat2 = Concatenate()([avg_pool2, upconv2])
    
    # Final 1x1 convolution to consolidate features
    conv5 = Conv2D(512, (1, 1), activation='relu')(concat2)
    
    # Flatten and fully connected layer
    flatten = Flatten()(conv5)
    fc = Dense(10, activation='softmax')(flatten)
    
    # Model creation
    model = Model(inputs=input_layer, outputs=fc)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Get the model
model = dl_model()