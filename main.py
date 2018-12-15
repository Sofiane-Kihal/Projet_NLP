from functions import *

name_file = str(input("Entrez le nom du fichier de texte :\n"))

#On ouvre le fichier contenant le texte
f_text = open(name_file+'.txt','r+')

#On ouvre et stocke les fichiers contenant les solutions (ils vont nous permettre d'évaluer notre application)
f_characters = open(name_file+'_CHARACTERS.txt','r+')
f_relationships = open(name_file+'_RELATIONSHIPS.txt','r+')
file_characters = f_characters.read()
file_relationships = f_relationships.read()

#On stocke le texte original et on retire ce qui pourrait fausser l'analyse des fonctions de la librairie nltk/stanfordcorenlp
text = f_text.read()
text = text.replace("’\n","\n")

#On initialise nos variables
CHARACTERS = []
DEPENDENCIES = [[]]*len(text)
TOKENS = []

#RELATIONSHIPS contient le résultat final
RELATIONSHIPS = {}
INCOMPLETE_RELATIONSHIP = {}

LINKS = ['son','father','mother','daughter','cousin','siblings','husband','wife','companion','mate','spouses','brother','sister','friend','girlfriend','boyfriend', 'uncle','aunt','nephew','niece','friends']
LINK_CORRESPONDANCE = {
    'son' : 'parent',
    'father' : 'child',
    'mother' : 'child',
    'daughter' : 'parent',
    'cousin' : 'cousin',
    'siblings' : 'siblings',
    'husband' : 'wife',
    'wife' : 'husband',
    'spouses' : 'spouses',
    'brother' : 'siblings',
    'sister' : 'siblings',
    'friend' : 'friend',
    'girlfriend' : 'couple',
    'boyfriend' : 'couple',
    'uncle' : 'nephew/niece',
    'aunt' : 'nephew/niece',
    'nephew' : 'uncle/aunt',
    'niece' : 'uncle/aunt',
    'mate' : 'mate',
    'companion' : 'companion',
    'friends' : 'friend'
}

#On extrait les personnages du texte, on les stocke dans CHARACTERS et on initialise notre dictionnaire RELATIONSHIPS
CHARACTERS = extract_NE(text,CHARACTERS)
init_relationships(RELATIONSHIPS,CHARACTERS)

#On procède à du traitement sur le texte original :
# on découpe le texte par mot et on remplace les occurrences faisant référence aux personnages telles que 'him/his/her/etc..' par le nom des personnages et on reconstitue le texte.
# On découpe en tokens de phrase ce nouveau texte.
# Enfin, on tokénise par mot chaque phrase et on la stocke dans TOKENS où on a une liste de liste, chacune des sous-listes contenant une phrase tokenizée en mots .
tmp_text = text.split()
text = replace_by_NE(tmp_text,CHARACTERS)
text_tokens = sent_tokenize(text)

for line in text_tokens :
    TOKENS.append(nlp.word_tokenize(line))

#On fait l'analyse des dépendances grammaticales et on les stocke dans DEPENDENCIES.
parsing = nlp.dependency_parse(text)
extract_dependencies(text_tokens,DEPENDENCIES)

#On remplit RELATIONSHIPS par les relations qu'on a réussi à extraire à l'aide de nos méthodes
fill_relationships(DEPENDENCIES,TOKENS,LINKS,RELATIONSHIPS, INCOMPLETE_RELATIONSHIP)

#On crée le symétrique des relations qu'on a extraite (voir la fonction corresponding_RELATIONSHIPS pour plus d'informations)
corresponding_RELATIONSHIPS = make_correspondance(RELATIONSHIPS,LINK_CORRESPONDANCE)
#corresponding_INCOMPLETE_RELATIONSHIP = make_correspondance(INCOMPLETE_RELATIONSHIP,LINK_CORRESPONDANCE)

#On fusionne RELATIONSHIPS et son symétrique pour obtenir la totalité des relations
merge_dictionnary(RELATIONSHIPS,corresponding_RELATIONSHIPS)
#merge_dictionnary(INCOMPLETE_RELATIONSHIP,corresponding_INCOMPLETE_RELATIONSHIP)

#On affiche le resultat

print("LES RELATIONS ENTRE LES PERSONNAGES DU TEXTE :")
for elt in RELATIONSHIPS :
    print(elt,' :', RELATIONSHIPS[elt])

#On évalue le résultat
print("EVALUATION :")
sol_CHARACTERS, sol_RELATIONSHIPS = makefile_solutions(file_characters,file_relationships)
accuracy(sol_CHARACTERS,sol_RELATIONSHIPS,RELATIONSHIPS,LINK_CORRESPONDANCE)
