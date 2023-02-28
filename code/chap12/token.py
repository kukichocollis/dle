import nltk
from nltk.tokenize import word_tokenize

nltk.download()
# nltk.download("treebank")
# nltk.download('punkt')


text = "This is a dog." # ③
print(word_tokenize(text))

tokens = word_tokenize("Hello World!, This is a dog.")

print(tokens)

# 문자나 숫자인 경우에만 단어를 리스트에 추가한다. 
words = [word for word in tokens if word.isalpha()]
print(words)

#
words2 = [word for word in tokens]
print(words2)
