MACHINE LEARNING MODELS :
    1- cannot use stop_words neither stemming and lemmatisation
    2- i use TF-IDF 
    3- inaccurate result are predicted 
    4- je ne dois pas traduire tous en un seul language

    problem : ML models are sensitive to ---> 
        - not well balanced data
        - use bag of word then no context
        - Too many useless/innexact words because of darja
        - cannot pre-process because of darja 
        - many languages 

    solution :
    - make a balanced data
    - make the model understand the context 
    - make the model know the similiraity between similars words
    - language should not be a problem

    ---> so use transformers ---> Bert / Roberta / CAMeL BERT / DistilBERT
        how to choose { CAMeL BERT : trained for arabe language , mBert : good but roberta is trained more , DistilBERT : good mais concu pour anglais principalement }


TRANSFORMERS MODELS :
    roberta ! 

    how to improve the accuracy of roberta :
        - normalize and standerize and clean the data the BEST way possible 
        - find the best combination of parameters
        - balance the data after analysing the problem in each label


A SAVOIR :
    Accuracy = (bonnes prédictions) / (total)
    Precision = vrai prediction / (vrai prediction + fausse prediction)
    Recall = vrai / (vrai + oubli)
    F1 = 2 * (precision * recall) / (precision + recall)      <-- vrai important si dataset desequilibre 



A FAIRE :
POUR NIVEAU 1 :
   - augementer un peu la taille de la class 4 : Suggestion ( not crucial )
POUR NIVEAU 2 :
    - rearange the parameters ( done ! )
    - probleme dans la class 3 : Réclamation service , regler la confusion (il confend class 3 avec 0 et 4) 
            ------> “force attention” sur classe 3 on modifiant les weights , donc le recall de la classe 3 devrait augmente
            ------> il faut eloigner les mots cles entre les class et supp les phrases qui peuvent etre avoir plusieurs classification
            ------> augementer un peu la taille de la class 3 et 4 ( not crucial )
POUR SENTIMENT ANALYSIS :
    - trop desequilibre , je dois reduire la taille des cas neagtif et utiliser class weight 
    - toujours probleme , surtout mauvais resultat 
    -----> donc data nest pas faite pour les sentiments 
    - utiliser Dziribirt only for sentiment (deja essaye) 



conflit avec la class 3 : Réclamation service 
    confendu avec Pannes et coupures générales , ces mot cle :{تسڨمونا, ماكاش , كونيكسيو , ثقيلة , انترنت , ADSL ,...} en resume internet coupe ou faible
    confendu avec Espace Client + My Idoom , ces mot cle {الزبون , ...} 
    --> donc augmenter cette class et se focaliser sur les mots cle a apart cela (done)
    --> + regler quelque erreur (done)


(same for the suggestion class in the first level)
    


    
