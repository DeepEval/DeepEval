input_layer = Input(shape=(32, 32, 3))

conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu')(maxpool1)
maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu')(maxpool2)
maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3)



conv4 = Conv2D(256, (3, 3), activation='relu')(maxpool3)
dropout = Dropout(0.5)(conv4)
conv5 = Conv2D(128, (3, 3), activation='relu')(dropout)


conv6 = Conv2D(64, (3, 3), activation='relu')(conv5)
upsample1 = ConvTranspose2D(64, (3, 3), activation='relu')(conv6)

conv7 = Conv2D(64, (3, 3), activation='relu')(upsample1)
upsample2 = ConvTranspose2D(64, (3, 3), activation='relu')(conv7)

conv8 = Conv2D(128, (3, 3), activation='relu')(upsample2)
upsample3 = ConvTranspose2D(128, (3, 3), activation='relu')(conv8)



output = Conv2D(10, (1, 1), activation='softmax')(upsample3)


model = Model(inputs=input_layer, outputs=output)