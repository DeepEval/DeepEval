import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.optimizers import Adam

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Function to create a functional model
def dl_model():
    # First block for feature extraction
    input_layer = Input(shape=x_train[0].shape)

    # Split input into three groups for depthwise separable convolutions
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Depthwise separable convolutions with different kernel sizes
    depthwise1 = Conv2D(64, (1, 1), activation='relu', use_bias=False)(split1[0])
    depthwise2 = Conv2D(64, (3, 3), activation='relu', use_bias=False)(split1[1])
    depthwise3 = Conv2D(64, (5, 5), activation='relu', use_bias=False)(split1[2])
    
    # Batch normalization
    bn1 = tf.keras.layers.BatchNormalization()(depthwise1)
    bn2 = tf.keras.layers.BatchNormalization()(depthwise2)
    bn3 = tf.keras.layers.BatchNormalization()(depthwise3)
    
    # Concatenate the outputs
    concatenated = concatenate([bn1, bn2, bn3])
    
    # Second block for feature extraction
    branch1 = Conv2D(32, (1, 1), activation='relu')(concatenated)
    branch2 = Conv2D(32, (3, 3), activation='relu')(concatenated)
    branch3 = MaxPooling2D(pool_size=(1, 7))(concatenated)  # 1x7 max pooling
    branch4 = Conv2D(32, (7, 1), activation='relu')(concatenated)  # 7x1 convolution
    branch5 = Conv2D(32, (3, 3), activation='relu')(branch4)  # followed by 3x3 conv
    
    # Average pooling
    avg_pool = Flatten()(MaxPooling2D(pool_size=(3, 3))(branch5))
    
    # Concatenate all branches
    concat_output = concatenate([branch1, branch2, branch3, avg_pool])
    
    # Fully connected layers for classification
    output = Dense(10, activation='softmax')(concat_output)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and train the model
model = dl_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)