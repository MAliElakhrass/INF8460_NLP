
"""
Questions 1.1.6 à 1.1.8 : calcul de différentes statistiques sur un corpus.

Sauf mention contraire, toutes les fonctions renvoient un nombre (int ou float).
Dans toutes les fonctions de ce fichier, le paramètre `corpus` désigne une liste de phrases tokenizées, par exemple :
>>> corpus = [
    ["Alice", "est", "là"],
    ["Bob", "est", "ici"]
]
"""
import preprocess_corpus as pre
import nltk


def count_tokens(corpus):
    """
    Renvoie le nombre de mots dans le corpus
    """
    counter = 0

    for sentence in corpus:
        for word in sentence:
            counter += 1

    return counter


def count_types(corpus):
    """
    Renvoie le nombre de types (mots distincts) dans le corpus
    """
    corpus1D = []
    for sentence in corpus:
        sentence = tuple(sentence)
        corpus1D.append(sentence)
    set(corpus1D)

    return len(corpus1D)


def get_most_frequent(corpus, n):
    """
    Renvoie les n mots les plus fréquents dans le corpus, ainsi que leurs fréquences

    :return: list(tuple(str, float)), une liste de paires (mot, fréquence)
    """
    corpus1D = []
    for i in corpus:
        for j in i:
            corpus1D.append(j)
    fdist = nltk.FreqDist(corpus1D)

    return fdist.most_common(n)


def get_token_type_ratio(corpus):
    """
    Renvoie le ratio nombre de tokens sur nombre de types
    """

    return count_tokens(corpus) / count_types(corpus)


def count_lemmas(corpus):
    """
    Renvoie le nombre de lemmes distincts
    """
    lemmas = pre.lemmatize(corpus)

    return count_types(lemmas)


def count_stems(corpus):
    """
    Renvoie le nombre de racines (stems) distinctes
    """
    stems = pre.stem(corpus)

    return count_types(stems)


def explore(corpus):
    """
    Affiche le résultat des différentes fonctions ci-dessus.

    Pour `get_most_frequent`, prenez n=15

    >>> explore(corpus)
    Nombre de tokens: 5678
    Nombre de types: 890
    ...
    Nombre de stems: 650

    """

    print("Nombre de tokens: " + str(count_tokens(corpus)))
    print("Nombre de types: " + str(count_types(corpus)))
    print("15 plus frequents")
    print(get_most_frequent(corpus, 15))
    print("Ratio: " + str(get_token_type_ratio(corpus)))
    print("Nombre de lemmes distincts: " + str(count_lemmas(corpus)))
    print("Nombre de racines (stems) distinctes: " + str(count_stems(corpus)))
    

if __name__ == "__main__":
    """
    Ici, appelez la fonction `explore` sur `shakespeare_train` et `shakespeare_test`. Quand on exécute le fichier, on 
    doit obtenir :

    >>> python explore_corpus
    -- shakespeare_train --
    Nombre de tokens: 5678
    Nombre de types: 890
    ...

    -- shakespeare_test --
    Nombre de tokens: 78009
    Nombre de types: 709
    ...
    """

    print("-- shakespeare_train --")
    corpus_shakespeare_train = pre.read_and_preprocess("data/shakespeare_train.txt")
    explore(corpus_shakespeare_train)

    print("-- shakespeare_test --")
    corpus_shakespeare_test = pre.read_and_preprocess("data/shakespeare_test.txt")
    explore(corpus_shakespeare_test)    
