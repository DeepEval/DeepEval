import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def residual_block(x):
    conv_1 = layers.Conv2D(64, (3, 3), padding='same')(x)
    act_1 = layers.Activation('relu')(conv_1)
    conv_2 = layers.Conv2D(64, (1, 1), padding='same')(act_1)
    add_1 = layers.Add()([x, conv_2])
    act_2 = layers.Activation('relu')(add_1)
    max_pool_1 = layers.MaxPool2D()(act_2)
    drop_1 = layers.Dropout(0.5)(max_pool_1)
    return drop_1

def dl_model():
  # Main Pathway
  inputs = keras.Input(shape=(28, 28, 1))
  conv_1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
  act_1 = layers.Activation('relu')(conv_1)
  conv_2 = layers.Conv2D(64, (1, 1), padding='same')(act_1)
  add_1 = layers.Add()([inputs, conv_2])
  act_2 = layers.Activation('relu')(add_1)
  max_pool_1 = layers.MaxPool2D()(act_2)
  drop_1 = layers.Dropout(0.5)(max_pool_1)

  # Branch Pathway
  branch_conv_1 = layers.Conv2D(64, (3, 3), padding='same')(max_pool_1)
  branch_act_1 = layers.Activation('relu')(branch_conv_1)
  branch_conv_2 = layers.Conv2D(64, (1, 1), padding='same')(branch_act_1)

  # Fusion
  add_2 = layers.Add()([drop_1, branch_conv_2])
  act_3 = layers.Activation('relu')(add_2)

  # Global Average Pooling
  gap_1 = layers.GlobalAveragePooling2D()(act_3)

  # Fully Connected Layer
  outputs = layers.Dense(10, activation='softmax')(gap_1)

  # Model Definition
  model = Model(inputs=inputs, outputs=outputs)
  return model