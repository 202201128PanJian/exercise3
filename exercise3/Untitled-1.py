import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, FreqDist
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

moby_dick_file = nltk.corpus.gutenberg.open('melville-moby_dick.txt')
moby_dick_text = moby_dick_file.read()

tokens = nltk.word_tokenize(moby_dick_text)

stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

pos_tags = nltk.pos_tag(filtered_tokens)

pos_freq = FreqDist([tag[1] for tag in pos_tags])
top_pos = pos_freq.most_common()
print("Parts of Speech frequency:")
for pos, freq in top_pos:
    print(f"{pos}: {freq}")

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token, pos=pos) for token, pos in pos_tags[:20]]
print("Lemmatized tokens:", lemmatized_tokens)

pos_freq.plot(30, cumulative=False)
plt.title("POS Frequency Distribution")
plt.xlabel("Parts of Speech")
plt.ylabel("Frequency")
plt.show()