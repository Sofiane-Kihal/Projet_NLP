#from stanfordcorenlp import StanfordCoreNLP
#from nltk.tokenize import sent_tokenize, word_tokenize
#import re
from functions import *

#nlp = StanfordCoreNLP('stanford-corenlp-full-2018-10-05')

file = open('TheLionKing.txt','r+')
text = file.read()
text = text.replace("’\n","\n")
PERSON = []
DEPENDENCY = [[]]*len(text)
TOKENS = []
RELATIONSHIP = {}
LINKS = ['son','father','mother','daughter','cousin','siblings','husband','wife','spouses','brother','sister','friend','girlfriend','boyfriend', 'uncle','aunt','nephew','niece']
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
    'niece' : 'uncle/aunt'
}

# EXTRACTION DES ENTITES NOMMEES

PERSON = extract_NE(text,PERSON)
init_relationship(RELATIONSHIP,PERSON)

# CREATION DE LA LISTE TOKENS
tmp_text = text.split()
text = replace_by_NE(tmp_text,PERSON)
text_tokens = sent_tokenize(text)

for line in text_tokens :
    TOKENS.append(nlp.word_tokenize(line))

# EXTRACTION DES DEPENDANCES GRAMMATICALES
extract_dependencies(text_tokens,DEPENDENCY)

# REMPLISSAGE DU DICTIONNAIRE 'RELATIONSHIP' QUI CONTIENT LES RELATIONS ENTRE LES PERSONNAGES
fill_relationship(DEPENDENCY,TOKENS,LINKS,RELATIONSHIP)

# CREATION DU SYMETRIQUE DE 'RELATIONSHIP' ET FUSION DES DEUX DICTIONNAIRES
corresponding_RELATIONSHIP = make_correspondance(RELATIONSHIP,LINK_CORRESPONDANCE)
merge_dictionnary(RELATIONSHIP,corresponding_RELATIONSHIP)

print(RELATIONSHIP)