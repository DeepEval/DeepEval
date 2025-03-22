import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Lambda, Add, concatenate, Dense, Flatten
from tensorflow.keras.optimizers import Adam


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0


def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First block: Dual-path structure
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x_branch = x
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(3, (1, 1), activation='sigmoid')(x)  # Restore the number of channels to match the input
    
    # Branch path
    x_branch = Conv2D(64, (3, 3), activation='relu')(x_branch)
    
    # Combine paths
    x = Add()([x, x_branch])
    
    # Second block: Feature extraction using depthwise separable convolutional layers
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
    
    x1 = Conv2D(64, (1, 1), activation='relu', padding='same')(split1[0])
    x1 = DepthwiseConv2D((3, 3), activation='relu')(x1)
    x1 = Conv2D(64, (1, 1), activation='relu', padding='same')(x1)
    
    x2 = Conv2D(64, (3, 3), activation='relu')(split1[1])
    x2 = DepthwiseConv2D((3, 3), activation='relu')(x2)
    x2 = Conv2D(64, (1, 1), activation='relu', padding='same')(x2)
    
    x3 = Conv2D(64, (5, 5), activation='relu')(split1[2])
    x3 = DepthwiseConv2D((5, 5), activation='relu')(x3)
    x3 = Conv2D(64, (1, 1), activation='relu', padding='same')(x3)
    
    # Concatenate the outputs
    x = concatenate([x1, x2, x3])
    
    # Flatten and add fully connected layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


model = dl_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)