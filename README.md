# ViterbiAlgorithm
A first-order HMM (Hidden Markov Model) for part of speech tagging (POS) developed in python. This includes; 
1) counting occurrences of one part of speech following another in a training corpus, 
2) counting occurrences of words together with parts of speech in a training corpus, 
3) relative frequency estimation with smoothing, 
4) finding the best sequence of parts of speech for a list of words in the test corpus, according to an HMM model with smoothed probabilities, 
5) computing the accuracy, that is, the percentage of parts of speech that is guessed correctly.

For running;
1. run the HMM.py to get the accuracy of the viterbi and the greedy best path algotithm
2. run the 'language_comparison.py' to obtain the results of language comparisons.
3. Make sure the 5 UD tree banks are available.
