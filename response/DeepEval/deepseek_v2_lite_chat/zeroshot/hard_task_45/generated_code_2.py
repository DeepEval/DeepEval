import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Functional model definition
def dl_model():
    # First block for feature extraction
    input_layer = Input(shape=x_train.shape[1:])
    
    split1 = Lambda(lambda x: tf.split(x, [1, 1, 1], axis=-1))(input_layer)
    x1 = split1[0]  # 1x1 conv
    x2 = split1[1]  # 3x3 conv
    x3 = split1[2]  # 5x5 conv
    
    # Separate depthwise separable convolutions
    x1 = DepthwiseConv2D((1, 1), activation='relu')(x1)
    x2 = DepthwiseConv2D((3, 3), activation='relu')(x2)
    x3 = DepthwiseConv2D((5, 5), activation='relu')(x3)
    
    # Concatenate outputs from all three groups
    x = Concatenate()([x1, x2, x3])
    
    # Second block for feature refinement
    split2 = Lambda(lambda x: tf.split(x, [1, 1, 1], axis=-1))(x)
    
    x1 = split2[0]  # 1x1 conv
    x2 = split2[1]  # 3x3 conv
    x3 = split2[2]  # max pooling
    
    x1 = DepthwiseConv2D((1, 1), activation='relu')(x1)
    x2 = DepthwiseConv2D((3, 3), activation='relu')(x2)
    x3 = DepthwiseConv2D((3, 3), activation='relu')(x3)
    
    x4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x3)  # 1x1 conv
    
    # Concatenate all branches
    x = Concatenate()([x1, x2, x3, x4])
    
    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)  # Assuming 10 classes
    
    # Build the model
    model = Model(input_layer, output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create the model
model = dl_model()