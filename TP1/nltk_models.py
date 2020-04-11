"""
Questions 1.4.1 à 1.6.2 : modèles de langue NLTK

Dans ce fichier, on rassemble les fonctions concernant les modèles de langue NLTK :
- entraînement d'un modèle de langue sur un corpus d'entraînement, avec ou sans lissage
- évaluation d'un modèle sur un corpus de test
- génération de texte suivant un modèle de langue

Pour préparer les données avant d'utiliser un modèle, on pourra utiliser
>>> ngrams, words = padded_everygram_pipeline(n, corpus)
>>> vocab = Vocabulary(words, unk_cutoff=k)

Lors de l'initialisation d'un modèle, il faut passer une variable order qui correspond à l'ordre du modèle n-gramme,
et une variable vocabulary de type `Vocabulary`.

On peut ensuite entraîner le modèle avec la méthode `model.fit(ngrams)`
"""
import numpy as np
import preprocess_corpus
import mle_ngram_model
import matplotlib.pyplot as plt
import random

from nltk.lm.models import MLE, Laplace, Lidstone
from nltk.lm.vocabulary import Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline


def train_LM_model(corpus, model, n, gamma=None, unk_cutoff=2):
    """
    Entraîne un modèle de langue n-gramme NLTK de la classe `model` sur le corpus.

    :param corpus: list(list(str)), un corpus tokenizé
    :param model: un des éléments de (MLE, Lidstone, Laplace)
    :param n: int, l'ordre du modèle
    :param gamma: float or None, le paramètre gamma (pour `model=Lidstone` uniquement). Si model=Lidstone, alors cet
    argument doit être renseigné
    :param unk_cutoff: le seuil au-dessous duquel un mot est considéré comme inconnu et remplacé par <UNK>
    :return: un modèle entraîné
    """
    ngrams, words = padded_everygram_pipeline(n, corpus)
    vocab = Vocabulary(words, unk_cutoff=unk_cutoff)
    
    if model == MLE:
        model = MLE(n, vocab)
        model.fit(ngrams)
    elif model == Lidstone and gamma != None:
        model = Lidstone(gamma, n, vocab)
        model.fit(ngrams)
    elif model == Laplace:
        model = Laplace(n, vocab)
        model.fit(ngrams)
    
    return model


def evaluate(model, corpus):
    """
    Renvoie la perplexité du modèle sur une phrase de test.

    :param model: nltk.lm.api.LanguageModel, un modèle de langue
    :param corpus: list(list(str)), une corpus tokenizé
    :return: float
    """
    corpus_ngram = mle_ngram_model.extract_ngrams(corpus, n)
    corpus_ngram = [gram for sentence in corpus_ngram for gram in sentence] 
    return model.perplexity(corpus_ngram)


def evaluate_gamma(gamma, train, test, n):
    """
    Entraîne un modèle Lidstone n-gramme de paramètre `gamma` sur un corpus `train`, puis retourne sa perplexité sur un
    corpus `test`.

    [Cette fonction est déjà complète, on ne vous demande pas de l'écrire]

    :param gamma: float, la valeur de gamma (comprise entre 0 et 1)
    :param train: list(list(str)), un corpus d'entraînement
    :param test: list(list(str)), un corpus de test
    :param n: l'ordre du modèle
    :return: float, la perplexité du modèle sur train
    """
    lm = train_LM_model(train, Lidstone, n, gamma=gamma)
    return evaluate(lm, test)


def generate(model, n_words, text_seed=None, random_seed=None):
    """
    Génère `n_words` mots à partir du modèle.

    Vous utiliserez la méthode `model.generate(num_words, text_seed, random_seed)` de NLTK qui permet de générer
    du texte suivant la distribution de probabilité du modèle de langue.

    Cette fonction doit renvoyer une chaîne de caractère détokenizée (dans le cas de Trump, vérifiez que les # et les @
    sont gérés); si le modèle génère un symbole de fin de phrase avant d'avoir fini, vous devez recommencer une nouvelle
    phrase, jusqu'à avoir produit `n_words`.

    :param model: un modèle de langue entraîné
    :param n_words: int, nombre de mots à générer
    :param text_seed: tuple(str), le contexte initial. Si aucun text_seed n'est précisé, vous devrez utiliser le début
    d'une phrase, c'est à dire respectivement (), ("<s>",) ou ("<s>", "<s>") pour n=1, 2, 3
    :param random_seed: int, la seed à passer à la méthode `model.generate` pour pouvoir reproduire les résultats. Pour
    ne pas fixer de seed, il suffit de laisser `random_seed=None`
    :return: str
    """
    tweet = ""
    required = False
    while (not required):
        tweet = model.generate(n_words, text_seed, random_seed)

        if len(tweet) == n_words and '<s>' not in tweet and '</s>' not in tweet:
            required = True
            new_tweet = []
            i = 0
            for element in tweet:
                if i != n_words - 1:
                    if element == '@' or element == '#':
                        new_tweet.append(tweet[i] + tweet[i + 1])
                        tweet.remove(tweet[i])
                    else:
                        new_tweet.append(element)
                    i += 1
                else:
                    new_tweet.append(element)
    
    return new_tweet


if __name__ == "__main__":
    """
    Vous aurez ici trois tâches à accomplir ici :
    
    1)
    Dans un premier temps, vous devez entraîner des modèles de langue MLE et Laplace pour n=1, 2, 3 à l'aide de la 
    fonction `train_MLE_model` sur le corpus `shakespeare_train` (question 1.4.2). Puis vous devrez évaluer vos modèles 
    en mesurant leur perplexité sur le corpus `shakespeare_test` (question 1.5.2).
    
    2)
    Ensuite, on vous demande de tracer un graphe représentant le perplexité d'un modèle Lidstone en fonction du paramètre 
    gamma. Vous pourrez appeler la fonction `evaluate_gamma` (déjà écrite) sur `shakespeare_train` et `shakespeare_test` 
    en faisant varier gamma dans l'intervalle (10^-5, 1) (question 1.5.3). Vous utiliserez une échelle logarithmique en 
    abscisse et en ordonnée.
    
    Note : pour les valeurs de gamma à tester, vous pouvez utiliser la fonction `numpy.logspace(-5, 0, 10)` qui renvoie 
    une liste de 10 nombres, répartis logarithmiquement entre 10^-5 et 1.
    
    3)
    Enfin, pour chaque n=1, 2, 3, vous devrez générer 2 segments de 20 mots pour des modèles MLE entraînés sur Trump.
    Réglez `unk_cutoff=1` pour éviter que le modèle ne génère des tokens <UNK> (question 1.6.2).
    """
    corpus_shakespeare_train = preprocess_corpus.read_and_preprocess("data/shakespeare_train.txt")
    corpus_shakespeare_test = preprocess_corpus.read_and_preprocess("data/shakespeare_test.txt")
   
    for n in range(1, 4):
        print("n= " + str(n))
        mle_model = train_LM_model(corpus_shakespeare_train, MLE, n) 
        laplace_model = train_LM_model(corpus_shakespeare_train, Laplace, n)
        print("Perplexite MLE= " + str(evaluate(mle_model, corpus_shakespeare_test)))
        print("Perplexite Laplace= " + str(evaluate(laplace_model, corpus_shakespeare_test)))
        
        lindston_perplexity = []
        gamma_val = []
        for gamma in np.logspace(-5, 0, 10):
            gamma_val.append(gamma)
            lindston_perplexity.append(evaluate_gamma(gamma, corpus_shakespeare_train, corpus_shakespeare_test, n))
        
        plt.subplots()
        plt.plot(gamma_val, lindston_perplexity)
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Perplexité en fonction du paramètre gamma")
        plt.xlabel("gamma")
        plt.ylabel("perplexité")
        plt.savefig("output/plot_" + str(n) + ".png")
        plt.close()


    # Trump
    corpus_trump = preprocess_corpus.read_and_preprocess("data/trump.txt")
    for n in range(1, 4):
        print("n= " + str(n))
        if (n == 1):
            text_seed = ()
        elif (n == 2):
            text_seed = ("<s>",)
        elif (n == 3):
            text_seed = ("<s>", "<s>")

        mle_model = train_LM_model(corpus_trump, MLE, n, unk_cutoff=1)

        print(generate(mle_model, 20, text_seed))
        print(generate(mle_model, 20, text_seed))
