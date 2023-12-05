'''
Paraphasia Detection w/ engineered features extracted from SB transcripts
Input: transcripts, feature representations from SB model
Output: Paraphasia detection (linear classifier)


Features:
-Duration: Forced alignment
-DTW of character representations (CTC output)
'''