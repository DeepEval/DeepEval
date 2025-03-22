import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Add, concatenate, Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

def dl_model():
  # Input layer
  inputs = Input(shape=(32, 32, 3))

  # Feature extraction
  x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)

  # Feature enhancement
  x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
  x = Dropout(0.5)(x)
  x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)

  # Feature restoration
  x = UpSampling2D(size=(2, 2))(x)
  x = Conv2DTranspose(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
  x = Add()([x, Conv2DTranspose(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(x))])
  x = UpSampling2D(size=(2, 2))(x)
  x = Conv2DTranspose(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
  x = Add()([x, Conv2DTranspose(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(x))])
  x = UpSampling2D(size=(2, 2))(x)
  x = Conv2DTranspose(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)

  # Classification
  outputs = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax', padding='same')(x)

  # Model construction
  model = Model(inputs=inputs, outputs=outputs)

  return model