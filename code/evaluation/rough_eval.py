from rouge import Rouge

def evaluate_summary(generated, reference):
    rouge = Rouge()
    scores = rouge.get_scores(generated, reference)
    return scores
