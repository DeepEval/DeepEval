import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize the input data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First block: Split along the channel axis and apply separable convolutions
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    
    conv1x1 = Conv2D(32, (1, 1), activation='relu')(split1[0])
    conv3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(split1[1])
    conv5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(split1[2])
    
    concat1 = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Second block: Multiple branches for enhanced feature extraction
    branch1 = Conv2D(64, (3, 3), activation='relu')(concat1)
    
    branch2 = Conv2D(64, (1, 1), activation='relu')(concat1)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    
    branch3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(concat1)
    
    concat2 = Concatenate()([branch1, branch2, branch3])
    
    # Global average pooling and fully connected layer
    gap = GlobalAveragePooling2D()(concat2)
    outputs = Dense(10, activation='softmax')(gap)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()