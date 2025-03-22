import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, AveragePooling2D

def dl_model():
    # Path 1: Two convolution blocks followed by average pooling
    input_layer = Input(shape=(32, 32, 3))
    conv_block1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv_block2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv_block1)
    avg_pool1 = AveragePooling2D(pool_size=(2, 2))(conv_block2)
    
    # Path 2: Single convolution layer followed by flattening
    conv_path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    flatten_path2 = Flatten()(conv_path2)
    
    # Add the features from both paths
    add_layer = Add()([avg_pool1, flatten_path2])
    
    # Fully connected layer for classification
    dense1 = Dense(units=128, activation='relu')(add_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint and early stopping
checkpoint = keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[checkpoint, early_stopping])