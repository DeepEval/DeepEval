import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Data augmentation
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(x_train)

# Define the model
def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Three paths of separable convolution and batch normalization
    def separable_conv_block(input_layer, filters, kernel_size):
        conv = Conv2D(filters, kernel_size, padding='same', use_bias=False)(input_layer)
        conv = tf.keras.layers.DepthwiseSeparableConv2D(kernel_size, depth_multiplier=1)(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Activation('relu')(conv)
        return conv
    
    # Split input into three groups and pass through different blocks
    path1 = separable_conv_block(input_layer, 32, (1, 1))
    path2 = separable_conv_block(input_layer, 64, (3, 3))
    path3 = separable_conv_block(input_layer, 64, (5, 5))
    
    # Concatenate outputs of the three paths
    merged1 = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], x[2]], axis=-1))(
        [path1, path2, path3]
    )
    
    # Block 2: Four parallel paths of convolution and pooling
    def parallel_conv_path(input_layer, filters, kernel_size, pool_size=(2, 2)):
        conv = Conv2D(filters, kernel_size, padding='same')(input_layer)
        conv = tf.keras.layers.MaxPooling2D(pool_size)(conv)
        conv = tf.keras.layers.Activation('relu')(conv)
        return conv
    
    # Four paths as described
    path1 = parallel_conv_path(merged1, 128, (1, 1))
    path2 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(input_layer)
    path3 = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], x[2]], axis=-1))(
        [tf.keras.layers.Lambda(lambda x: tf.concat([x[0], x[2]], axis=-1))(
            [tf.keras.layers.Conv2D(filters, (1, 3), padding='same')(input_layer), tf.keras.layers.Conv2D(filters, (3, 1), padding='same')(input_layer)]
        ),
        tf.keras.layers.Conv2D(filters, (1, 3), padding='same')(input_layer),
        tf.keras.layers.Conv2D(filters, (3, 1), padding='same')(input_layer)
    ]
    )
    path4 = parallel_conv_path(input_layer, 128, (1, 1))
    path4 = tf.keras.layers.Conv2D(256, (3, 3))(path4)
    path4 = tf.keras.layers.Lambda(lambda x: tf.concat([x[0], x[2]], axis=-1))(
        [tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(path4), tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(path4)]
    )
    
    # Concatenate outputs of the four paths
    merged2 = tf.keras.layers.Concatenate()([path1, path2, path3, path4])
    
    # Flatten and fully connected layers
    output_layer = Flatten()(merged2)
    predictions = Dense(10, activation='softmax')(output_layer)
    
    # Model
    model = Model(inputs=input_layer, outputs=predictions)
    
    # Compile model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and return the model
model = dl_model()
model.summary()