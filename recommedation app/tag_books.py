import pandas as pd

def get_emotion_tag(title):
    title = title.lower()
    if any(word in title for word in ['love', 'romance', 'heart', 'happy']):
        return 'happy'
    elif any(word in title for word in ['war', 'death', 'cry', 'alone']):
        return 'sad'
    elif any(word in title for word in ['revenge', 'dark', 'blood', 'fight']):
        return 'angry'
    else:
        return 'neutral'

df = pd.read_csv('BookDataset/Bookz.csv')
df['emotion_tag'] = df['Title'].apply(get_emotion_tag)
df.to_csv('BookDataset/Bookz_tagged.csv', index=False)
print("✅ Bookz_tagged.csv created successfully.")
