import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

sentences = [
			'I love my dog',
			'I love my cat',
			'You love my dog!',
			'Do you think my dog is amazing?',
			]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

test_data = ['i really love my dog', 'my dog loves my manatee']

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)
print(word_index)

print(sequences)