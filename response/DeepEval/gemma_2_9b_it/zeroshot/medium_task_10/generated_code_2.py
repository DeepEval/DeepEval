import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Level 1: Initial Convolution and Basic Block
    x = layers.Conv2D(16, kernel_size=3, padding='same', activation='relu')(inputs)

    # Basic Block
    x = layers.BasicBlock(filters=16)(x)

    # Level 2: Two Residual Blocks
    x = layers.ResidualBlock(filters=16)(x) 
    x = layers.ResidualBlock(filters=16)(x)

    # Level 3: Feature Fusion and Global Branch
    x = layers.Conv2D(16, kernel_size=1, padding='same')(x)  
    x = layers.add([x, layers.Conv2D(16, kernel_size=1, padding='same')(x)]) 

    # Average Pooling and Fully Connected Layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# Define BasicBlock and ResidualBlock
class BasicBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size=3, padding='same')
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return layers.add([x, inputs])

class ResidualBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.main_path = BasicBlock(filters)
        self.branch = layers.Conv2D(filters, kernel_size=1, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, inputs):
        main_path = self.main_path(inputs)
        branch = self.bn(self.branch(inputs))
        return layers.add([main_path, branch])