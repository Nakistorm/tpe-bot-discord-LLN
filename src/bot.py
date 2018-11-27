
""" Imports de librairie """
import nltk
import sys
from sys import exit
# TODO : dès que le programme fonctionne, ajouter l'import Discord : https://github.com/Rapptz/discord.py

""" Cette variable va modeliser la base de connaissances du bot """
training = []
tweets = []
chemin_e = 'C:\\Users\\naelg_jm9bezd\\Desktop\\tpe-bot-discord-LLN\\connaissances_bot\\entree\\'
chemin_r = 'C:\\Users\\naelg_jm9bezd\\Desktop\\tpe-bot-discord-LLN\\connaissances_bot\\reception\\'

""" Formate les jeux de données pour le classifieur """
def set_training(path, intent):
    f = open(path, 'rU')
    samples = [(l.strip(), intent) for l in f]
    return samples

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

""" on ajoute à l'entrainement nos phrases pour que le bot s'approprie des concepts """
# TODO : modifier les chemins des fichiers .txt (les créer aussi si nécessaire). Ce serait utile si vous vous mettiez d'accord sur un chemin commun (rappelez-vous que vous travaillez en équipe) 
training += set_training(chemin_r+'blagues.txt', 'blagues')
training += set_training(chemin_r+'ca_va.txt', 'ca va')
training += set_training(chemin_r+'note_tpe.txt', 'note tpe')
training += set_training(chemin_r+'politique.txt', 'politique')
training += set_training(chemin_r+'questions_prenom.txt', 'questions prenom')
training += set_training(chemin_r+'salut.txt', 'salut')

for (words, sentiment) in training:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment))

word_features = get_word_features(get_words_in_tweets(tweets))

training_set = nltk.classify.apply_features(extract_features, tweets)
""" classification et entrainement actif du bot avec les jeux de données fournis """
classifier = nltk.NaiveBayesClassifier.train(training_set)

""" test de reconnaissance avec fichier test """
user_inputs = []
if len(sys.argv) > 1:
    tweetfile = sys.argv[1]
    user_inputs.append(tweetfile)
        
for user_input in user_inputs:
  valued = classifier.classify(extract_features(user_input.split()))
  limit = min(40,len(user_input))
  print (user_input[:limit] + '\n : \t' + valued)

classifier.show_most_informative_features() 
exit()

