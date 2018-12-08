from stanfordcorenlp import StanfordCoreNLP
from nltk.tokenize import sent_tokenize, word_tokenize
import re

nlp = StanfordCoreNLP('stanford-corenlp-full-2018-10-05')

def extract_NE(text, PERSON):
    """Stocke dans une liste (PERSON) les entités nommées d'un texte (text)"""
    NE = nlp.ner(text)
    for elt in NE:
        if 'PERSON' in elt:
            PERSON.append(elt[0])

    return (list(set(PERSON)))


def init_relationship(RELATIONSHIP, PERSON):
    """ Initialise un dictionnaire (RELATIONSHIP) sur la base d'une liste (PERSON) d'entité nommée. """
    for p in PERSON:
        RELATIONSHIP[p] = []


def extract_dependencies(text_tokens, DEPENDENCY):
    """
    Extrait les dépendances 'nmod:poss' et 'appos' au sein d'un texte découpé par phrases (text_tokens) et les insère dans DEPENDENCY.

    arguments :

    text_tokens : liste contenant des listes, chaque 'sous-liste' contient elle une phrase du texte d'origine.
    DEPENDENCY : tableau qui contient dans chaque colonne les dépendances grammaticales liées à une phrase.

    """
    nb_line = 0
    for line in text_tokens:
        parsing = nlp.dependency_parse(line)
        for elt in parsing:

            if ('nmod:poss' in elt):
                DEPENDENCY[nb_line] = DEPENDENCY[nb_line] + [(elt)]

            elif ('appos' in elt):
                DEPENDENCY[nb_line] = DEPENDENCY[nb_line] + [(elt)]
        nb_line += 1
    nlp.close()


def make_relation(dep, dependencies, nb_line):
    """
    Crée un quadruplé qui correspond à une relation valide sous la forme (personne1,relation,personne2,ligne du texte
    où la relation a été identifiée).

        Arguments :
        dep : une dépendance de la forme ('nature de la dépendance',mot1,mot2)
        dependencies : liste de dépendances contenues dans la même ligne de texte que 'dep'
        nb_line : numéro de la ligne du texte où la dépendance a été récupérée

    """

    # On vérifie que la dépendance est bien de type 'nmod:poss'
    if (dep[0] == 'nmod:poss'):

        # Si il n'y a pas d'autres dépendance dans la phrase, alors on ne pourra pas extraire les 2 personnages
        # On renvoie donc un quadruplé de type ('NaN',relation,personnage2,numéro de la ligne dans le texte)
        if (dependencies == []):
            return (('NaN', dep[1], dep[2], nb_line))

        # Sinon...
        else:
            # ..on va chercher dans le reste des dépendances...
            for elt in dependencies:
                # ..une apposition.
                if (elt[0] == 'appos'):

                    # Si l'élement (elt[1]) (qui est un personnage) de l'apposition est identique à celui de notre nmod:poss (dep[1])...
                    # ..cela signifie qu'ils sont liés par un élément commun (qui est la relation dans le texte)
                    if (dep[1] == elt[1]):
                        # Alors on retourne le quadruplé (personnage1,relation,personnage2,numéro de la ligne dans le texte)
                        return ((elt[2], dep[1], dep[2], nb_line))

    return ((0, 0, 0))


def replace_by_NE(text_split, PERSON):
    """Remplace toutes les occurences de la liste 'to_replace' par l'entité nommée à laquelle
    elles font référence dans le texte """

    to_replace = ['he', 'she', 'his', 'him', 'her', 'He', 'She', 'His', 'Her', 'Him']
    result = ""
    tmp_NE = ""

    for elt in text_split:

        tmp = re.sub('\W+', '', elt)

        if (tmp in PERSON):
            tmp_NE = tmp
            # result = result + elt + " "

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


def fill_relationship(DEPENDENCY, TOKENS, LINKS, RELATIONSHIP):
    """
    Remplis le dictionnaire RELATIONSHIP avec les relations qui ont pu être extraites à l'aide de la fonction
    'make_relation()'

    arg :

    DEPENDENCY : Le tableau de dépendances dont on a parlé dans les fonctions précédentes

    TOKENS : Une liste de liste. Chaque sous-liste contient une phrase du texte qui a été découpée en 'token' dont
    les indices des mots correspondent à ceux présents dans les dépendances du tableau DEPENDENCY. Cette liste va
    nous permettre de transformer un indice en mot pour pouvoir avoir une sortie compréhensible.

    RELATIONSHIP : Dans la description de la fonction

    LINKS : Liste qui contient les relations possibles entre des personnages
    """
    nb_line = 0
    tmp = []
    encoded_relationship = []
    id_NaN = 0

    # Pour toutes les dépendances qu'on a recueillis dans le tableau DEPENDENCY on créer les relations qui existent
    # en utilisant la fonction make_relation et on les stocke dans la liste tmp
    for elt in DEPENDENCY:
        try:
            tmp.append(make_relation(elt[0], elt[1:], nb_line))
            nb_line += 1

        except:
            nb_line += 1

    # Pour chaque élément de la liste tmp, on remplit la liste "encoded_relationship" de tous les éléments validés
    for t in tmp:
        if (t != (0, 0, 0)):
            encoded_relationship.append(t)

    # Cette boucle va permettre de traduire une relation de type (1,mother,3,5) en (Gertrude,mother,Hamlet) et la stocker
    # dans le dictionnaire RELATIONSHIP
    for r in encoded_relationship:

        try:
            line = r[3]
            perso2 = r[0] - 1
            lien = r[1] - 1
            perso1 = r[2] - 1

            try:

                if ((TOKENS[line][lien]) in LINKS):
                    RELATIONSHIP[(TOKENS[line][perso1])] += [(TOKENS[line][lien], TOKENS[line][perso2])]

            except:

                if ((TOKENS[line][lien]) in LINKS):
                    RELATIONSHIP[(TOKENS[line][perso1])] = [(TOKENS[line][lien], TOKENS[line][perso2])]


        except:

            line = r[3]
            lien = r[1] - 1
            perso1 = r[2] - 1

            if ((TOKENS[line][lien]) in LINKS):
                RELATIONSHIP[str(r[0]) + str(id_NaN)] = [(TOKENS[line][lien], TOKENS[line][perso1])]

                id_NaN += 1


def make_correspondance(RELATIONSHIP, LINK_CORRESPONDANCE):
    """
    Créer un dictionnaire dans lequel on trouve la symétrie de chaque relation présente dans le dictionnaire RELATIONSHIP

    Par exemple :
    RELATIONSHIP : Hamlet : [(mother,Gertrude)] -> new_RELATIONSHIP : Gertrude : [(child,Hamlet)]
    (Hamlet a pour mère Gertrude) -> (Gertrude a pour fils Hamlet)

    """
    new_RELATIONSHIP = {}

    for elt in RELATIONSHIP:
        for i in RELATIONSHIP[elt]:

            try:
                new_RELATIONSHIP[i[1]] += [(LINK_CORRESPONDANCE[i[0]], elt)]

            except:

                new_RELATIONSHIP[i[1]] = [(LINK_CORRESPONDANCE[i[0]], elt)]
    return new_RELATIONSHIP


def merge_dictionnary(dict1, dict2):
    """Fusionne deux dictionnaire (va nous permettre de fusionner RELATIONSHIP et son symétrique)"""
    for elt in dict2:
        for i in dict2[elt]:
            try:
                dict1[elt] += [i]
            except:
                dict1[elt] = [i]