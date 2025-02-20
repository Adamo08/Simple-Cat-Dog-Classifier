import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from utilities import *

# 🔹 Initialization function
def initialize(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

# 🔹 Model function (Logistic Regression)
def model(X, W, b):
    Z = np.dot(X, W) + b
    Z = np.clip(Z, -500, 500)  # Prevent extreme values
    A = 1 / (1 + np.exp(-Z))
    return A

# 🔹 Log-Loss function
def log_loss(A, y):
    epsilon = 1e-8  # Small value to avoid log(0)
    A = np.clip(A, epsilon, 1 - epsilon)  
    return -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))

# 🔹 Gradient function
def gradient(X, y, A):
    dZ = A - y
    dW = np.dot(X.T, dZ) / X.shape[0]
    db = np.mean(dZ)
    return (dW, db)

# 🔹 Update function
def update(W, b, dW, db, alpha):
    W -= alpha * dW
    b -= alpha * db
    return (W, b)

# 🔹 Prediction function
def predict(X, W, b):
    A = model(X, W, b)
    return np.round(A)

# 🔹 Training function with Normal visualizations
def ANN_NORMAL(X, y, alpha, epochs):
    W, b = initialize(X)
    Loss = []
    Acc = []
    
    for i in tqdm(range(epochs)):
        A = model(X, W, b)
        loss = log_loss(A, y)
        
        if i % 10 == 0:
            Loss.append(loss)
            y_pred = predict(X, W, b)
            accuracy = accuracy_score(y, y_pred)
            Acc.append(accuracy)
        
        dW, db = gradient(X, y, A)
        W, b = update(W, b, dW, db, alpha)

    # 🎯 Predictions & Accuracy
    y_pred = predict(X, W, b)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # 📈 Stunning Loss + Accuracy Curve
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12, color='tab:red')
    ax1.plot(Loss, label='Loss', color='#FF6F61', linewidth=2, linestyle='dashed')
    ax1.scatter(range(0, len(Loss), len(Loss)//10), Loss[::len(Loss)//10], 
                color='blue', edgecolors='black', zorder=3)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, linestyle='dotted', alpha=0.6)
    
    # Create a second y-axis for Accuracy
    ax2 = ax1.twinx()  
    ax2.set_ylabel("Accuracy", fontsize=12, color='tab:blue')
    ax2.plot(Acc, label='Accuracy', color='tab:blue', linewidth=2)
    ax2.scatter(range(0, len(Acc), len(Acc)//10), Acc[::len(Acc)//10], 
                color='red', edgecolors='black', zorder=3)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    fig.suptitle("Training Loss and Accuracy Over Epochs", fontsize=16, fontweight='bold', color='#333333')
    fig.tight_layout()
    plt.show()

    # 🔥 Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False, linewidths=1, linecolor='black')
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold', color='#333333')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    return W, b


# 🔹 Training function with Live visualizations
def ANN_LIVE(X, y, alpha, epochs):
    W, b = initialize(X)
    Loss = []
    Acc = []

    # Enable interactive mode
    plt.ion()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    for i in tqdm(range(epochs)):
        A = model(X, W, b)
        loss = log_loss(A, y)

        if i % 10 == 0:
            Loss.append(loss)
            y_pred = predict(X, W, b)
            accuracy = accuracy_score(y, y_pred)
            Acc.append(accuracy)

            # Live Plotting
            ax1.clear()
            ax2.clear()

            ax1.set_xlabel("Epochs", fontsize=12)
            ax1.set_ylabel("Loss", fontsize=12, color='tab:red')
            ax1.plot(Loss, label='Loss', color='#FF6F61', linewidth=2, linestyle='dashed')
            ax1.scatter(range(len(Loss)), Loss, color='blue', edgecolors='black', zorder=3)
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax1.grid(True, linestyle='dotted', alpha=0.6)

            ax2.set_ylabel("Accuracy", fontsize=12, color='tab:blue')
            ax2.plot(Acc, label='Accuracy', color='tab:blue', linewidth=2)
            ax2.scatter(range(len(Acc)), Acc, color='red', edgecolors='black', zorder=3)
            ax2.tick_params(axis='y', labelcolor='tab:blue')

            fig.suptitle("Training Loss and Accuracy Over Epochs", fontsize=16, fontweight='bold', color='#333333')
            fig.tight_layout()
            plt.pause(0.01)  # Pause for a short time to update the figure

        dW, db = gradient(X, y, A)
        W, b = update(W, b, dW, db, alpha)

    plt.ioff()  # Disable interactive mode when training is done
    plt.show()

    # Final Confusion Matrix
    y_pred = predict(X, W, b)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False, linewidths=1, linecolor='black')
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold', color='#333333')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    return W, b




# 📥 Load data
X_train, y_train, X_test, y_test = load_data()

# 🔍 Show Sample Images
plt.figure(figsize=(16, 8))
for i in range(1, 10):
    plt.subplot(4, 5, i)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 🔄 Preprocessing: Reshape and Normalize
# Normalize pixel values to [0, 1]

"""
    Min-Max Normalisation
    X = (X - min(X)) / (max(X) - min(X))
    Black pixel: 0, White pixel: 255
    Min: 0, Max: 255
    X = X / 255.0
    
"""

X_train_reshape = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test_reshape = X_test.reshape(X_test.shape[0], -1) / 255.0

print("X_train_reshape shape:", X_train_reshape.shape)
print("X_test_reshape shape:", X_test_reshape.shape)

# 🔥 Train The ANN
"""
    Use ANN_NORMAL for Normal visualizations
    Use ANN_LIVE for Live visualizations
"""

W, b = ANN_LIVE(X_train_reshape, y_train, 0.01, 10000)
