import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Flatten, Dense, Concatenate
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model function
def dl_model():
    # Input placeholder
    inputs = Input(shape=(32, 32, 3))
    
    # Main path
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    x1 = split1[0]  # No change in the first group
    x2 = split1[1]  # Feature extraction with 3x3 conv
    x3 = split1[2]  # Additional 3x3 convolution
    
    # Branch path
    x = Conv2D(64, (1, 1), padding='same')(inputs)  # 1x1 conv layer
    
    # Concatenation of outputs from main path
    concat = Concatenate()([x1, x2, x3])
    
    # Further processing in branch path
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    
    # Combine main and branch paths
    x = tf.keras.layers.Add()([concat, x])
    
    # Flatten and fully connected layer for classification
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    
    return model

# Return the constructed model
model = dl_model()
print(model.summary())