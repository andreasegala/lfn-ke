from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
"""
    sulle metriche di scikit learn ci sono già queste funzioni
    
    sono stra easy da usare perchè una volta che abbiamo gli output basta fare:
    
        precision_score(y_true, y_pred)
        recall_score(y_true, y_pred)
        f1_score(y_true, y_pred)
        
    se le lunghezze di y_true e y_pred non coincidono non c'è problema, gestiscono il caso
    
    !!!NBNBNBNB!!! l'unico dettaglio è che non lavorano direttamente con stringhe ma con numeri
    c'è la classe LabelEncoder che può tornare utile:
    
        label_encoder = LabelEncoder()
        y_true_numeric = label_encoder.fit_transform(y_true_strings)
        y_pred_numeric = label_encoder.transform(y_pred_strings)
        
    altrimenti one hot encoding ma non so se possa tornare utile (la vedo complicata con questo tbh)
    
    idem per la precision recall curve:
    
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve')
        plt.show()
           
    ps. un ringraziamento speciale a ChatGPT per questi micro-snippet ahahah
"""