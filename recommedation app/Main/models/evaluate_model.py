import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# 1. मॉडल लोड करें
model_path = 'models/emotion_model.h5'  # अपना सही पाथ डालें
model = load_model(model_path)

# 2. Validation डेटा तैयार करें (अपना डेटा पाथ डालें)
val_data_dir = 'data/validation'  # उसी फ़ोल्डर को इस्तेमाल करें जो ट्रेनिंग में था
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(48, 48),  # अपने मॉडल के इनपुट शेप के अनुसार बदलें
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False  # Confusion Matrix के लिए जरूरी
)

# 3. मॉडल की accuracy चेक करें
val_loss, val_acc = model.evaluate(val_generator)
print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")

# 4. Confusion Matrix के लिए predictions निकालें
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes
class_names = list(val_generator.class_indices.keys())

# 5. Confusion Matrix प्लॉट करें
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')  # इमेज सेव होगी
plt.show()

# 6. Classification Report प्रिंट करें
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))