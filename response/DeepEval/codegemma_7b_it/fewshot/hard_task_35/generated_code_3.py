import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_branch(input_tensor, num_filters):
  x = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(input_tensor)
  x = layers.MaxPooling2D()(x)
  x = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(x)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dense(num_filters)(x)
  x = layers.Reshape((num_filters, 1, 1))(x)
  return x

def dl_model():
  input_tensor = layers.Input(shape=(32, 32, 3))

  branch_1_output = create_branch(input_tensor, 32)
  branch_2_output = create_branch(input_tensor, 64)

  merged_outputs = layers.Concatenate()([branch_1_output, branch_2_output])
  x = layers.Flatten()(merged_outputs)
  x = layers.Dense(10, activation='softmax')(x)

  model = keras.Model(inputs=input_tensor, outputs=x)
  return model

# Compile and train the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64)