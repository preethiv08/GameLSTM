
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical
from keras.optimizers import Adam

dataset = pd.read_excel("C:\\Users\\Dell\\Downloads\\AI data.xlsx")

text_data = dataset['RULES OF THE GAME'].astype(str).values.tolist()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
total_words = len(tokenizer.word_index) + 1

# Create input sequences using the tokenized text
input_sequences = []
for line in text_data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = to_categorical(y, num_classes=total_words)

# LSTM Model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_length-1))
model.add(LSTM(50, dropout=0.2))  # Added dropout
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.fit(X, y, epochs=30, batch_size=32, verbose=1)



# Generate game concepts
def generate_game_concepts(seed_text, num_concepts=10, max_words=100, temperature=0.5):
    game_concepts = set()  
    for _ in range(num_concepts):
        input_text = seed_text
        for _ in range(max_words):  
            token_list = tokenizer.texts_to_sequences([input_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
            predicted_probs = model.predict(token_list)[0]
            predicted_word_index = np.random.choice(len(predicted_probs), p=predicted_probs)
            predicted_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_word_index:
                    predicted_word = word
                    break
            input_text += " " + predicted_word
            if predicted_word in [".", "?", "!"]:
                break
        game_concepts.add(input_text)

    return list(game_concepts)


seed_text = "In a mysterious world"
generated_concepts = generate_game_concepts(seed_text, num_concepts=len(dataset), max_words=100)

dataset_reset_index = dataset.reset_index(drop=True)

output_df = dataset_reset_index[['MECHANIC', 'OBJECTIVE OF THE GAME', 'USP', 'LEVEL FAIL CONDITION', 'RULES OF THE GAME']].copy()

if len(generated_concepts) == len(output_df):
    output_df['Generated Concepts'] = generated_concepts
    output_df.to_excel('generated_concepts_with_columns.xlsx', index=False)
    print("Generated concepts with columns saved to 'generated_concepts_with_columns.xlsx'")
else:
    print("Lengths of generated concepts and DataFrame do not match. Please check your data.")
