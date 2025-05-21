import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load dataset + Decoding
train_images_path = r"C:\Users\rupes\Downloads\gzip\gzip\emnist-balanced\emnist-balanced-train-images-idx3-ubyte\emnist-balanced-train-images-idx3-ubyte"
train_labels_path = r"C:\Users\rupes\Downloads\gzip\gzip\emnist-balanced\emnist-balanced-train-labels-idx1-ubyte\emnist-balanced-train-labels-idx1-ubyte"
test_images_path = r"C:\Users\rupes\Downloads\gzip\gzip\emnist-balanced\emnist-balanced-test-images-idx3-ubyte\emnist-balanced-test-images-idx3-ubyte"
test_labels_path = r"C:\Users\rupes\Downloads\gzip\gzip\emnist-balanced\emnist-balanced-test-labels-idx1-ubyte\emnist-balanced-test-labels-idx1-ubyte"
mapping_path = r"C:\Users\rupes\Downloads\gzip\gzip\emnist-balanced\emnist-balanced-mapping.txt"

train_images = idx2numpy.convert_from_file(train_images_path)
train_labels = idx2numpy.convert_from_file(train_labels_path)
test_images = idx2numpy.convert_from_file(test_images_path)
test_labels = idx2numpy.convert_from_file(test_labels_path)

# Load mapping
mapping = {}
with open(mapping_path, 'r') as f:
    for line in f:
        label, ascii_code = map(int, line.split())
        mapping[label] = chr(ascii_code)

# Preprocess data
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_images = np.transpose(train_images, (0, 2, 1))
test_images = np.transpose(test_images, (0, 2, 1))
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

num_classes = 47
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)
datagen.fit(train_images)

# Build
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

#Train model
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=128, subset='training'),
    validation_data=datagen.flow(train_images, train_labels, batch_size=128, subset='validation'),
    epochs=25,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Predict and visualize samples
predictions = model.predict(test_images[:5])
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    true_label = mapping[np.argmax(test_labels[i])]
    pred_label = mapping[predicted_labels[i]]
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis('off')
plt.tight_layout()
plt.show()

model.save('emnistmodel.keras')
print("Model saved as 'emnistmodel.keras")
