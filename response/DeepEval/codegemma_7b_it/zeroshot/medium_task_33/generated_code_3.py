from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, concatenate, Flatten, Dense, Activation

def dl_model():
  # Define input layer
  inputs = Input(shape=(32, 32, 3))

  # Split image into three channel groups
  r, g, b = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

  # Feature extraction for each channel group
  r_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(r)
  r_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(r_conv)
  r_conv = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(r_conv)

  g_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(g)
  g_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(g_conv)
  g_conv = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(g_conv)

  b_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(b)
  b_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(b_conv)
  b_conv = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(b_conv)

  # Concatenate outputs from all three groups
  merged = concatenate([r_conv, g_conv, b_conv])

  # Average pooling
  avg_pool = MaxPooling2D(pool_size=(2, 2))(merged)

  # Flatten
  flattened = Flatten()(avg_pool)

  # Fully connected layers
  fc1 = Dense(units=128, activation='relu')(flattened)
  fc2 = Dense(units=64, activation='relu')(fc1)
  fc3 = Dense(units=10, activation='softmax')(fc2)

  # Create model
  model = Model(inputs=inputs, outputs=fc3)

  return model

# Instantiate the model
model = dl_model()