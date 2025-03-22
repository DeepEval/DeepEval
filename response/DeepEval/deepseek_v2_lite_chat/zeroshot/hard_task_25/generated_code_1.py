import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ZeroPadding2D, Conv2DTranspose, InputLayer
from tensorflow.keras.layers import AveragePooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the input tensor
    inputs = Input(shape=(32, 32, 3))

    # Main Path
    x = Conv2D(64, (1, 1), activation='relu')(inputs)  # 1x1 conv
    x = Conv2D(64, (3, 3), activation='relu')(x)       # 3x3 conv

    x1 = MaxPooling2D(pool_size=(3, 3))(inputs)         # downsample
    x1 = Conv2D(64, (3, 3), activation='relu')(x1)    # 3x3 conv
    x1 = UpSampling2D(size=(3, 3))(x1)                # upscale

    x2 = AveragePooling2D(pool_size=(3, 3))(inputs)     # downsample
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)   # 3x3 conv
    x2 = UpSampling2D(size=(3, 3))(x2)               # upscale

    x3 = inputs                                       # inputs as third branch
    x = Concatenate()([x, x3])                         # concatenate

    # Branch Path
    x = Conv2D(64, (1, 1), activation='relu')(x)      # 1x1 conv
    x = Conv2D(64, (3, 3), activation='relu')(x)      # 3x3 conv
    x = Conv2D(64, (3, 3), activation='relu')(x)      # 3x3 conv

    # Final Output
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Get the model
model = dl_model()