from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():

  # Input layer
  img_input = Input(shape=(28, 28, 1))

  # Main path
  x = Conv2D(32, (5, 5), activation='relu')(img_input)
  x = Dropout(0.25)(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Conv2D(64, (5, 5), activation='relu')(x)
  x = Dropout(0.25)(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Conv2D(128, (5, 5), activation='relu')(x)
  x = Dropout(0.25)(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Conv2D(10, (3, 3), activation='relu')(x)
  x = Dropout(0.25)(x)

  # Branch path
  y = Conv2D(10, (3, 3), activation='relu')(img_input)

  # Combine outputs
  combined = concatenate([x, y])

  # Flatten and fully connected layer
  combined = Flatten()(combined)
  combined = Dense(128, activation='relu')(combined)
  combined = Dropout(0.5)(combined)

  # Output layer
  output = Dense(10, activation='softmax')(combined)

  # Create model
  model = Model(inputs=img_input, outputs=output)

  return model