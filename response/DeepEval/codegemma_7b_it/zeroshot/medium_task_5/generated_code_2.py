from tensorflow.keras import layers, models

def dl_model():
  # Input layer
  input_img = layers.Input(shape=(32, 32, 3))

  # Main path
  x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
  x = layers.MaxPooling2D((2, 2))(x)
  x = layers.Conv2D(64, (3, 3), activation='relu')(x)
  main_output = layers.MaxPooling2D((2, 2))(x)

  # Branch path
  y = layers.Conv2D(64, (3, 3), activation='relu')(input_img)
  branch_output = layers.MaxPooling2D((2, 2))(y)

  # Combine outputs
  combined = layers.Add()([main_output, branch_output])

  # Flatten and fully connected layers
  combined = layers.Flatten()(combined)
  combined = layers.Dense(512, activation='relu')(combined)
  combined = layers.Dense(10, activation='softmax')(combined)

  # Model definition
  model = models.Model(inputs=input_img, outputs=combined)

  return model