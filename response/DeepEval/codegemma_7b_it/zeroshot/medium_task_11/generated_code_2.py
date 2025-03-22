from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, concatenate, multiply, Conv2DTranspose, ReLU, BatchNormalization, Dropout, Flatten

def dl_model():
  # Input layer
  img = Input(shape=(32, 32, 3))

  # Initial convolutional layer
  conv_init = Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(img)

  # Channel attention pathway
  avg_pool_ch = GlobalAveragePooling2D()(conv_init)
  dense_avg_pool_ch = Dense(64, activation="relu")(avg_pool_ch)
  dense_avg_pool_ch_sig = Dense(64, activation="sigmoid")(dense_avg_pool_ch)

  max_pool_ch = GlobalMaxPooling2D()(conv_init)
  dense_max_pool_ch = Dense(64, activation="relu")(max_pool_ch)
  dense_max_pool_ch_sig = Dense(64, activation="sigmoid")(dense_max_pool_ch)

  ch_attention = multiply([dense_avg_pool_ch_sig, dense_max_pool_ch_sig])
  ch_attention = Reshape((1, 1, 64))(ch_attention)
  ch_attention = multiply([ch_attention, conv_init])

  # Spatial attention pathway
  avg_pool_sp = GlobalAveragePooling2D()(ch_attention)
  dense_avg_pool_sp = Dense(64, activation="relu")(avg_pool_sp)
  dense_avg_pool_sp_sig = Dense(32, activation="sigmoid")(dense_avg_pool_sp)

  max_pool_sp = GlobalMaxPooling2D()(ch_attention)
  dense_max_pool_sp = Dense(64, activation="relu")(max_pool_sp)
  dense_max_pool_sp_sig = Dense(32, activation="sigmoid")(dense_max_pool_sp)

  sp_attention = concatenate([dense_avg_pool_sp_sig, dense_max_pool_sp_sig])
  sp_attention = Dense(64, activation="relu")(sp_attention)
  sp_attention = Dense(32, activation="sigmoid")(sp_attention)

  # Fused feature map
  fused_feature_map = multiply([sp_attention, ch_attention])

  # Spatial features
  avg_pool_sp_f = GlobalAveragePooling2D()(fused_feature_map)
  max_pool_sp_f = GlobalMaxPooling2D()(fused_feature_map)

  # Concatenate spatial features
  spatial_features = concatenate([avg_pool_sp_f, max_pool_sp_f])

  # Channel features
  avg_pool_ch_f = GlobalAveragePooling2D()(fused_feature_map)
  max_pool_ch_f = GlobalMaxPooling2D()(fused_feature_map)

  # Concatenate channel and spatial features
  fusion_features = concatenate([avg_pool_ch_f, max_pool_ch_f, spatial_features])

  # Fully connected layer
  output = Dense(10, activation="softmax")(fusion_features)

  # Model
  model = keras.Model(img, output)

  return model