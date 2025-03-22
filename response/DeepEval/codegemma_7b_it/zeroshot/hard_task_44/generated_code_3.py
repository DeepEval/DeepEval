from tensorflow.keras import Model, Input, layers
from tensorflow.keras import backend as K

def dl_model():
  # Input layer
  inputs = Input(shape=(32, 32, 3))

  # Block 1
  x = Lambda(lambda x: K.split(x, 3, axis=3))(inputs)
  x = [
      layers.Conv2D(64, (1, 1), padding='same', activation='relu')(branch)
      for branch in x
  ]
  x = [layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch)
       for branch in x
  ]
  x = [layers.Conv2D(64, (5, 5), padding='same', activation='relu')(branch)
       for branch in x
  ]
  x = [layers.Dropout(0.25)(branch) for branch in x]
  x = layers.concatenate(x)

  # Block 2
  branches = [
      layers.Conv2D(64, (1, 1), padding='same', activation='relu'),
      layers.Conv2D(64, (1, 1), padding='same', activation='relu'),
      layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
      layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
      layers.Conv2D(64, (1, 1), padding='same', activation='relu'),
  ]

  outputs = []
  for branch in branches:
    outputs.append(branch(x))

  # Feature fusion
  x = layers.concatenate(outputs)

  # Output layer
  x = layers.Flatten()(x)
  x = layers.Dense(64, activation='relu')(x)
  outputs = layers.Dense(10, activation='softmax')(x)

  # Model definition
  model = Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model