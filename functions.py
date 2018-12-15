from stanfordcorenlp import StanfordCoreNLP
from nltk.tokenize import sent_tokenize, word_tokenize
import re

nlp = StanfordCoreNLP('stanford-corenlp-full-2018-10-05')


def extract_NE(text, CHARACTERS):
    """ Extrait les entités nommées d'un texte (text) et les stocke dans une liste (CHARACTERS)."""
    NE = nlp.ner(text)
    for elt in NE:
        if 'PERSON' in elt:
            CHARACTERS.append(elt[0])

    return (list(set(CHARACTERS)))


def init_relationships(RELATIONSHIPS, CHARACTERS):
    """ Initialise un dictionnaire (RELATIONSHIPS) qui contiendra les relations entre personnages sur la base d'une liste (CHARACTERS) d'entité nommée. """
    for p in CHARACTERS:
        RELATIONSHIPS[p] = []


def extract_dependencies(text_tokens, DEPENDENCIES):
    """
    Extrait les dépendances 'nmod:poss', 'appos', 'coumpund' et 'dep' au sein d'un texte découpé par phrases (text_tokens) et les insère dans DEPENDENCIES.

    arguments :

    text_tokens : liste contenant des listes, chaque 'sous-liste' contient elle une phrase du texte d'origine.
    DEPENDENCIES : tableau qui contient dans chaque colonne les dépendances grammaticales présentes à une phrase.

    """
    nb_line = 0
    for line in text_tokens:
        parsing = nlp.dependency_parse(line)
        for elt in parsing:

            if ('nmod:poss' in elt or 'appos' in elt or 'nmod' in elt or 'compound' in elt or 'dep' in elt):
                DEPENDENCIES[nb_line] = DEPENDENCIES[nb_line] + [(elt)]

        nb_line += 1
    nlp.close()


def make_relation(dep, dependencies, nb_line):
    """
    Renvoie une liste encoded_relationships contenant des quadruplés qui correspondent à des relations valides sous la forme (personne1,relation,personne2,ligne du texte
    où la relation a été identifiée)

        Arguments :

        dep : une dépendance de la forme ('nature de la dépendance',mot1,mot2)

        dependencies : liste de dépendances contenues dans la même ligne de texte que 'dep'

        nb_line : numéro de la ligne du texte où la dépendance a été récupérée

    """
    # On initialise la liste
    encoded_relationships = []

    # On vérifie que la dépendance est bien de type 'nmod:poss'
    if (dep[0] == 'nmod:poss' or dep[0] == 'nmod'):

        # Si il n'y a pas d'autres dépendance dans la phrase, alors on ne pourra pas extraire les 2 personnages
        # On ajoute donc un quadruplé de type ('NaN',relation,personnage2,numéro de la ligne dans le texte)
        if (dependencies == []):
            encoded_relationships.append(('NaN', dep[1], dep[2], nb_line))

        # Sinon...
        else:
            # ..on va chercher dans le reste des dépendances...
            for elt in dependencies:
                # ..une apposition.
                # Si l'élement (elt[1]) (qui est un personnage) de l'apposition est identique à celui de notre nmod:poss (dep[1])...
                # ..cela signifie qu'ils sont liés par un élément commun (qui est la relation dans le texte)
                if ((elt[0] == 'appos' and dep[0] == 'nmod:poss') and dep[1] == elt[1]):
                    # Alors on ajoute le quadruplé (personnage1,relation,personnage2,numéro de la ligne dans le texte)
                    encoded_relationships.append((elt[2], dep[1], dep[2], nb_line))

                # Le principe est le même pour les autres règles
                elif ((elt[0] == 'appos' and dep[0] == 'nmod') and dep[1] == elt[2]):
                    encoded_relationships.append((elt[1], dep[1], dep[2], nb_line))

                elif ((elt[0] == 'compound' and dep[0] == 'nmod:poss') and elt[1] == dep[1]):
                    encoded_relationships.append((dep[2], elt[2], dep[1], nb_line))

                elif ((elt[0] == 'dep' and dep[0] == 'nmod:poss') and elt[1] == dep[1]):
                    encoded_relationships.append((dep[2], elt[1], elt[2], nb_line))

    return (encoded_relationships)


def replace_by_NE(text_split, CHARACTERS):
    """Remplace toutes les occurences de la liste 'to_replace' par l'entité nommée à laquelle
    elles font référence dans le texte """

    to_replace = ['he', 'she', 'his', 'him', 'her', 'He', 'She', 'His', 'Her', 'Him']
    result = ""
    tmp_NE = ""

    for elt in text_split:

        tmp = re.sub('\W+', '', elt)

        if (tmp in CHARACTERS):
            tmp_NE = tmp

        if (elt in to_replace):

            # Ici on rajoute le suffixe "'s" quand on rencontre 'his' ou 'her' car sinon on risque de changer les dépendances grammaticales
            # Et l'analyseur de dépendance risque de ne pas détecter un nmod:poss ou une apposition

            if (elt == "his" or elt == "her"):
                result = result + tmp_NE + "'s'" + " "
            else:
                result = result + tmp_NE + " "

        else:
            result = result + elt + " "

    return result


def fill_relationships(DEPENDENCIES, TOKENS, LINKS, RELATIONSHIPS, INCOMPLETE_RELATIONSHIP):
    """
    Remplit le dictionnaire RELATIONSHIP avec les relations qui ont pu être extraites à l'aide de la fonction
    'make_relation()'

    arguments :

    DEPENDENCIES : Le tableau de dépendances décrit dans les fonctions précédentes

    TOKENS : Une liste de liste. Chaque sous-liste contient une phrase du texte qui a été découpée en 'token' dont
    les indices des mots correspondent aux indices des mots présents dans les dépendances du tableau DEPENDENCIES.
    Cette liste va nous permettre de transformer un indice du tableau DEPENDENCIES en mot pour pouvoir avoir une sortie compréhensible.

    RELATIONSHIPS : Dictionnaire qui va contenir toutes les relations entre personnages.

    LINKS : Liste qui contient le nom des relations possibles entre des personnages

    """
    nb_line = 0
    tmp = []
    encoded_relationships = []
    id_NaN = 0

    # Pour toutes les dépendances qu'on a recueilli dans le tableau DEPENDENCY on crée les relations qui existent
    # en utilisant la fonction make_relation et on les stocke dans la liste tmp
    for elt in DEPENDENCIES:
        try:
            for i in elt:
                for j in (make_relation(i, elt, nb_line)):
                    tmp.append(j)
            nb_line += 1

        except:
            nb_line += 1

    # Pour chaque élément de la liste tmp, on remplit la liste "encoded_relationships" de tous les éléments validés
    for t in tmp:
        encoded_relationships.append(t)

    # Cette boucle va permettre de traduire une relation de type (1,mother,3,5) en (Gertrude,mother,Hamlet) et la stocker
    # dans le dictionnaire RELATIONSHIPS
    for r in encoded_relationships:

        try:
            line = r[3]
            perso2 = r[0] - 1
            lien = r[1] - 1
            perso1 = r[2] - 1

            try:

                if ((TOKENS[line][lien]) in LINKS):
                    RELATIONSHIPS[(TOKENS[line][perso1])] += [(TOKENS[line][lien], TOKENS[line][perso2])]

            except:

                if ((TOKENS[line][lien]) in LINKS):
                    RELATIONSHIPS[(TOKENS[line][perso1])] = [(TOKENS[line][lien], TOKENS[line][perso2])]

        # S'il manque un élément dans la relation, on le stocke dans le dictionnaire correspondant
        except:

            line = r[3]
            lien = r[1] - 1
            perso1 = r[2] - 1

            try:
                if ((TOKENS[line][lien]) in LINKS):
                    INCOMPLETE_RELATIONSHIP[TOKENS[line][perso1]] += [(TOKENS[line][lien], str(r[0]) + str(id_NaN))]

                    id_NaN += 1
            except:

                if ((TOKENS[line][lien]) in LINKS):
                    INCOMPLETE_RELATIONSHIP[TOKENS[line][perso1]] = [(TOKENS[line][lien], str(r[0]) + str(id_NaN))]

                    id_NaN += 1


def make_correspondance(RELATIONSHIPS, LINK_CORRESPONDANCE):
    """
    Crée un dictionnaire dans lequel on trouve la symétrie de chaque relation présente dans le dictionnaire RELATIONSHIP

    Par exemple :
    RELATIONSHIPS : Hamlet : [(mother,Gertrude)] -> new_RELATIONSHIPS : Gertrude : [(child,Hamlet)]
    (Hamlet a pour mère Gertrude) -> (Gertrude a pour fils Hamlet)

    """
    new_RELATIONSHIPS = {}

    for elt in RELATIONSHIPS:
        for i in RELATIONSHIPS[elt]:

            try:
                new_RELATIONSHIPS[i[1]] += [(LINK_CORRESPONDANCE[i[0]], elt)]

            except:

                new_RELATIONSHIPS[i[1]] = [(LINK_CORRESPONDANCE[i[0]], elt)]
    return new_RELATIONSHIPS


def merge_dictionnary(dict1, dict2):
    """Fusionne deux dictionnaire (va nous permettre de fusionner RELATIONSHIP et son symétrique)"""
    for elt in dict2:
        for i in dict2[elt]:
            try:
                dict1[elt] += [i]
            except:
                dict1[elt] = [i]


def accuracy(sol_CHARACTERS, sol_RELATIONSHIPS, RELATIONSHIPS, LINK_CORRESPONDANCE):
    """Calcule la précision et le rappel du resultat de notre programme

    arguments :

    sol_CHARACTERS : une liste contenant tous les personnages du texte (conçue par l'utilisateur)
    sol_RELATIONSHIPS : une liste contenant toutes les relations entre les personnages du texte (conçue par l'utilisateur)
    RELATIONSHIPS : le dictionnaire résultant de l'application
    LINK_CORRESPONDANCE : dictionnaire contenant le symétrique de chaque nom de relation


    """
    good_EN = 0
    good_relationships = 0

    denominator_precision = 0
    result_precision = 0

    denominator_recall = len(sol_CHARACTERS) + len(sol_RELATIONSHIPS) * 2
    result_recall = 0

    tmp = []

    for elt in RELATIONSHIPS:
        if elt in sol_CHARACTERS:
            good_EN += 1

    for elt in sol_RELATIONSHIPS:

        try:
            tmp.append((elt[1], elt[2]) in RELATIONSHIPS[elt[0]])
            tmp.append((LINK_CORRESPONDANCE[elt[1]], elt[0]) in RELATIONSHIPS[elt[2]])

            good_relationships = sum(tmp)

        except KeyError:
            pass

    for elt in RELATIONSHIPS:
        denominator_precision += 1
        denominator_precision += len(RELATIONSHIPS[elt])

    result_precision = (good_EN + good_relationships) / denominator_precision

    result_recall = (good_EN + good_relationships) / denominator_recall

    print("Précision :", result_precision, '%')
    print("Rappel :", result_recall, '%')


def makefile_solutions(file_characters, file_relationships):
    """Transforme les entrées d'un fichier .txt en listes lisibles pour la fonction accuracy"""

    sol_CHARACTERS = file_characters.split("\n")

    sol_RELATIONSHIPS = []
    sol_RELATIONSHIPS_tmp = file_relationships.split("\n")

    for elt in sol_RELATIONSHIPS_tmp:
        sol_RELATIONSHIPS.append(tuple(elt.split(',')))

    return sol_CHARACTERS, sol_RELATIONSHIPS