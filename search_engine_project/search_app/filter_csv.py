import pandas as pd
import os
from views import preprocess_text
from transformers import pipeline, RobertaTokenizerFast, RobertaForSequenceClassification

# Initialize sentiment analysis pipeline
tokenizer = RobertaTokenizerFast.from_pretrained('arpanghoshal/EmoRoBERTa')
model = RobertaForSequenceClassification.from_pretrained('arpanghoshal/EmoRoBERTa', from_tf=True)
emotion_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Load data
current_dir = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(current_dir, 'collection', 'netflix_titles.csv'))

# Drop rows with NaN values in specified columns and reset index
data_cleaned = data.dropna(subset=data.columns[1:11]).reset_index(drop=True)

# Process descriptions and predict emotions
emotions = []
emotion_scores = []
for idx, row in data_cleaned.iterrows():
    description = preprocess_text(row['description'])
    prediction = emotion_pipeline(description)
    print(prediction)
    emotions.append(prediction)

# Add emotions to DataFrame
data_cleaned['emotion'] = emotions

# Save the modified DataFrame back to CSV
data_cleaned.to_csv(os.path.join(current_dir, 'collection', 'netflix_titles_processed.csv'), index=False)
