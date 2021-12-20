from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def flatten_list(unflat_list):
    return [item for sublist in unflat_list for item in sublist]

def get_metrics(final_preds, final_labels, prefix):
    final_preds = flatten_list(final_preds)
    final_labels = flatten_list(final_labels)
    
    macro_f1_score = f1_score(final_labels, final_preds, average="macro")
    macro_precision_score = precision_score(final_labels, final_preds, average="macro")
    macro_recall_score = recall_score(final_labels, final_preds, average="macro")
    accuracy = accuracy_score(final_labels, final_preds)

    return { prefix+'_f1': macro_f1_score, 
    prefix+'_precision': macro_precision_score, 
    prefix+'_recall': macro_recall_score, 
    prefix+'_accuracy': accuracy}

