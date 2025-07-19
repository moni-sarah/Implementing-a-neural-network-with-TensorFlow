import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess the Fashion MNIST dataset
def load_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)

# Build the neural network model
def build_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_data()
    model = build_model()
    model.fit(train_images, train_labels, epochs=10, batch_size=32)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"✅ Test accuracy: {test_acc:.4f}")
