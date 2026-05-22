facteur de priorite (numerique) :
    - type nievau 1 
    - type nievau 2
    - type sentiment 
    - date attente 


pour extraire le temps d'attente le message doit etre de type reclamation client 
how : 
    - utiliser pattern ( la plus simple et moins efficace )
    - finetuned un petit modele ( data and time are missing )
    - utiliser pattern pour ar/fr + dict pour derja ( besoin de detection de language )

cree une fonction qui nettoie le text 


pour faire Résumé automatique :

    # Extracte les entités nommées :
        pour anglais et francias j'utilise spacy , pour arabe stanza 

    # Extract la langauge :
        fonction de code-switching ( not efficent )
        finetuned un petit modele ( data and time are missing )
        utiliser un modele transformer de IbrahimAmin (best)

    # Extract date and time

    # the other are inputs from user : code --> departement , customer level , DOT , ACTEL's