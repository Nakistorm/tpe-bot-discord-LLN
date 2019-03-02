
""" Imports de librairie """
import nltk
import sys
import random
import discord
import asyncio
from sys import exit
# TODO : dès que le programme fonctionne, ajouter l'import Discord : https://github.com/Rapptz/discord.py

""" Cette variable va modeliser la base de connaissances du bot """
training = []
tweets = []
chemin_e = '..\\connaissances_bot\\entree\\'
chemin_r = '..\\connaissances_bot\\reception\\'


def set_training(path, intent):
    """ Formate les jeux de données pour le classifieur, retourne un tableau de (phrase, sentiment).
        Chaque ligne du fichier va être associée à l'intention correspondante.
        Exemple d'appel: set_training('C:\\monFichierContenantDesBonjour.txt','bonjour')

    Keyword arguments:
    path -- chemin du fichier texte
    intent -- label donné à l'intention correspondante voulue
    """
    f = open(path, 'r')
    samples = [(l.strip(), intent) for l in f]
    return samples

def get_words_in_tweets(tweets):
    """Récupère tous les mots d'un ensemble de (phrase, sentiment).
    
    Keyword arguments:
    tweets -- tableau contenant une liste de (phrase, sentiment)
    """
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    print('Voici les mots les plus utilisés :')
    print(wordlist.most_common(30))
    word_features = wordlist.keys()
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def flatten_list(multiList):
    out = []
    for item in multiList:
        if isinstance(item, list):
            out = out + flatten_list(item)
        else:
            out.append(item)
    return out

""" on ajoute à l'entrainement nos phrases pour que le bot s'approprie des concepts """
print('Alimentation de l\'entraînement du bot...')
training += set_training(chemin_r+'blagues.txt', 'blagues')
training += set_training(chemin_r+'ca_va.txt', 'ca_va')
training += set_training(chemin_r+'note_tpe.txt', 'note_tpe')
training += set_training(chemin_r+'politique.txt', 'politique')
training += set_training(chemin_r+'questions_prenom.txt', 'questions prenom')
training += set_training(chemin_r+'salut.txt', 'salut')

""" Filtrage de l'entrainement : on va décaler de training à tweets en ignorant les mots à une lettre,
	qui en général ne donne pas beaucoup d'information sur le sens de la phrase.
"""
for (words, sentiment) in training:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 2]
    tweets.append((words_filtered, sentiment))

""" on crée notre dictionnaire à partir des mots de notre jeu de données d'entrainement """
word_features = get_word_features(get_words_in_tweets(tweets))
training_set = nltk.classify.apply_features(extract_features, tweets)

""" classification et entrainement actif du bot avec les jeux de données fournis """
print('Le bot est prêt à s\'entrainer !')
classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Fait.')

class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')

    async def on_message(self, message):
        # on ne veut pas que le bot se réponde à lui-même
        if message.author.id == self.user.id:
            return
			
		# affichages pour déboguer
        print(self.user)
        print(message)
        print(message.content + '\n\n')

        self_ref = '<@' + str(self.user.id) + '>'
		
        if self_ref in message.content:
            msg_list = message.content.split(self_ref)
            msg = ''.join(msg_list)
            
            """ traitement du message reçu, séparation des mots et exécution """
            valued = classifier.classify(extract_features(msg.split()))
            limit = min(80,len(msg))
            print (msg[:limit] + '...\nIntention détectée: ' + valued)

            classifier.show_most_informative_features() 
            print('Précision attendue :')
            print(nltk.classify.accuracy(classifier, training_set))
            answers = set_training(chemin_e+valued+'.txt', valued)
            print(random.randint(0,len(answers)-1))
            bot_answer = answers[random.randint(0,len(answers)-1)]
            print(bot_answer)
            """"""
            await message.channel.send(bot_answer[0])
		
		# jeu : deviner un chiffre
        if message.content.startswith('$guess'):
            await message.channel.send('Guess a number between 1 and 10.')

            def is_correct(m):
                return m.author == message.author and m.content.isdigit()

            answer = random.randint(1, 10)

            try:
                guess = await self.wait_for('message', check=is_correct, timeout=5.0)
            except asyncio.TimeoutError:
                return await message.channel.send('Sorry, you took too long it was {}.'.format(answer))

            if int(guess.content) == answer:
                await message.channel.send('You are right!')
            else:
                await message.channel.send('Oops. It is actually {}.'.format(answer))

client = MyClient()

client.run('NTUxMzg2NTU4NjAzNzIyNzUy.D1wO0w.81GlQGsYQ_mRvbMftBdaXnIhMDw')