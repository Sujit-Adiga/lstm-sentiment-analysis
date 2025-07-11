import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
import nltk
import random

nltk.download('movie_reviews')
nltk.download('punkt')

# 1. Load dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# 2. Preprocess
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:5000]

def extract_features(words):
    words = set(words)
    return [1 if w in words else 0 for w in word_features]

X = [extract_features(doc) for doc, _ in documents]
y = [1 if category == "pos" else 0 for _, category in documents]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 3. Define LSTM Model
class SentimentLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SentimentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, seq_len=1, features)
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        out = self.fc(out)
        return self.sigmoid(out)

model = SentimentLSTM(input_size=5000, hidden_size=64)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. Training loop
for epoch in range(5):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. Evaluate
model.eval()
with torch.no_grad():
    preds = model(X_test)
    preds_cls = (preds > 0.5).float()
    acc = accuracy_score(y_test, preds_cls)
    print(f"Accuracy: {acc:.2f}")
