import nltk

"""
Questions 1.1.1 à 1.1.5 : prétraitement des données.
"""


def segmentize(raw_text):
    """
    Segmente un texte en phrases.

    >>> raw_corpus = "Alice est là. Bob est ici"
    >>> segmentize(raw_corpus)
    ["Alice est là.", "Bob est ici"]
    
    :param raw_text: str
    :return: list(str)
    """

    listSentence = nltk.sent_tokenize(raw_text)

    return listSentence


def tokenize(sentences):
    """
    Tokenize une liste de phrases en mots.

    >>> sentences = ["Alice est là", "Bob est ici"]
    >>> corpus = tokenize(sentences)
    >>> corpus_name
    [
        ["Alice", "est", "là"],
        ["Bob", "est", "ici"]
    ]

    :param sentences: list(str), une liste de phrases
    :return: list(list(str)), une liste de phrases tokenizées
    """

    if len(sentences) == 2:
        print(sentences)

    listWordsSentences = []
    for sentence in sentences:
        listWordsSentences.append(nltk.word_tokenize(sentence))
    
    if len(sentences) == 2:
        print(listWordsSentences)

    return listWordsSentences



def lemmatize(corpus):
    """
    Lemmatise les mots d'un corpus.

    :param corpus: list(list(str)), une liste de phrases tokenizées
    :return: list(list(str)), une liste de phrases lemmatisées
    """

    lemmzer = nltk.WordNetLemmatizer()
    listLemmesSentences = []

    for sentence in corpus:
        listLemmes = [lemmzer.lemmatize(word) for word in sentence]
        listLemmesSentences.append(listLemmes)

    return listLemmesSentences


def stem(corpus):
    """
    Retourne la racine (stem) des mots d'un corpus.

    :param corpus: list(list(str)), une liste de phrases tokenizées
    :return: list(list(str)), une liste de phrases stemées
    """
    stemmer = nltk.PorterStemmer()
    listStemSentences = []

    for sentence in corpus:
        listStem = [stemmer.stem(word) for word in sentence]
        listStemSentences.append(listStem)

    return listStemSentences


def read_and_preprocess(filename):
    """
    Lit un fichier texte, puis lui applique une segmentation et une tokenization.

    [Cette fonction est déjà complète, on ne vous demande pas de l'écrire]

    :param filename: str, nom du fichier à lire
    :return: list(list(str))
    """
    with open(filename, "r") as f:
        raw_text = f.read()
    return tokenize(segmentize(raw_text))


def write_result_file(corpus_name, tuple):
    """
    Ecrit dans un fichier texte le resultat du preprocesing

    :param path: str, path du fichier à ecrire
    """
    with open("output/" + corpus_name + "_phrases.txt", "w") as f:
        for sentence in tuple[0]:
            f.write(sentence + '\n')
    
    with open("output/" + corpus_name + "_mots.txt", "w") as f:
        for listWordsSentence in tuple[1]:
            for word in listWordsSentence:
                f.write(word + " ")
            f.write('\n')

    with open("output/" + corpus_name + "_lemmes.txt", "w") as f:
        for listLemmesSentence in tuple[2]:
            for lemme in listLemmesSentence:
                f.write(lemme + " ")
            f.write('\n')

    with open("output/" + corpus_name + "_stems.txt", "w") as f:
        for listStemSentence in tuple[3]:
            for stem in listStemSentence:
                f.write(stem + " ")
            f.write('\n')


def test_preprocessing(raw_text, sentence_id=0):
    """
    Applique à `raw_text` les fonctions segmentize, tokenize, lemmatize et stem, puis affiche le résultat de chacune
    de ces fonctions à la phrase d'indice `sentence_id`

    >>> trump = open("data/trump.txt", "r").read()
    >>> test_preprocessing(trump)
    Today we express our deepest gratitude to all those who have served in our armed forces.
    Today,we,express,our,deepest,gratitude,to,all,those,who,have,served,in,our,armed,forces,.
    Today,we,express,our,deepest,gratitude,to,all,those,who,have,served,in,our,armed,force,.
    today,we,express,our,deepest,gratitud,to,all,those,who,have,serv,in,our,arm,forc,.

    :param raw_text: str, un texte à traiter
    :param sentence_id: int, l'indice de la phrase à afficher
    :return: un tuple (sentences, tokens, lemmas, stems) qui contient le résultat des quatre fonctions appliquées à
    tout le corpus
    """

    sentences = segmentize(raw_text)
    tokens = tokenize(sentences)
    lemmes = lemmatize(tokens)
    stems = stem(tokens)

    print("All lemmes for sentence\n" + sentences[sentence_id])
    print(lemmes[sentence_id])
    print("All stems for sentence\n" + sentences[sentence_id])
    print(stems[sentence_id])

    return (sentences, tokens, lemmes, stems)


if __name__ == "__main__":
    """
    Appliquez la fonction `test_preprocessing` aux corpus `shakespeare_train` et `shakespeare_test`.

    Note : ce bloc de code ne sera exécuté que si vous lancez le script directement avec la commande :
    ```
    python preprocess_corpus.py
    ```
    """
    with open("data/shakespeare_train.txt", "r") as f:
        raw_text_shakespeare_train = f.read()
    tuple_shakespeare_train = test_preprocessing(raw_text_shakespeare_train)
    write_result_file("shakespeare_train", tuple_shakespeare_train)

    with open("data/shakespeare_test.txt", "r") as f:
        raw_text_shakespeare_test = f.read()
    tuple_shakespeare_test = test_preprocessing(raw_text_shakespeare_test)
    write_result_file("shakespeare_test", tuple_shakespeare_test)
