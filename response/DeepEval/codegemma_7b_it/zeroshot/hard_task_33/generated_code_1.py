from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def dl_model():

  # Load MNIST dataset
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
  x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255.0
  y_train = to_categorical(y_train, 10)
  y_test = to_categorical(y_test, 10)

  # Create input layer
  input_layer = layers.Input(shape=(28, 28, 1))

  # Define the block function
  def block(x):
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x)
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)
    return x

  # Create three branches
  branch_a = block(input_layer)
  branch_b = block(layers.AveragePooling2D(pool_size=(2, 2))(input_layer))
  branch_c = block(layers.MaxPooling2D(pool_size=(2, 2))(input_layer))

  # Concatenate branch outputs
  concat = layers.concatenate([branch_a, branch_b, branch_c])

  # Flatten and fully connected layer
  x = layers.Flatten()(concat)
  output = layers.Dense(units=10, activation='softmax')(x)

  # Create model
  model = models.Model(inputs=input_layer, outputs=output)

  return model