
import os
#from glob import glob # module for finding pathnames
#import pickle
import numpy as np
import tensorflow as tf


# ----------------------------------------------
#  P O S T E R I O R   H A N D L I N G
# ----------------------------------------------

# remove repeated tokens in sequence
# inputs (N,len)-shape tensor, returns list of lists
def remove_serial_duplicates(inp_seq):
    out_seq = [] # list of lists
    for i in range(inp_seq.shape[0]):
        q = []
        q.append(inp_seq[i][0].numpy()) # append first element
        for j in range(1,inp_seq.shape[1]):
            if inp_seq[i][j].numpy() != q[-1]:
                q.append(inp_seq[i][j].numpy())
        out_seq.append(q)
    return out_seq

# remove specific token from input sequence
# input and output is list of lists
def remove_specific_token(inp_seqs, tok):
    out_seqs = []
    for i in range(len(inp_seqs)):
        out_seqs.append([s for s in inp_seqs[i] if s != tok])
    return out_seqs


# ----------------------------------------------
#  E V A L U A T I O N
# ----------------------------------------------

# evaluate confusion matrix (TP, FN, FP) for a given keyword
# input data are batches
def cm_from_processed_seqs(array_true, array_pred, kw_token):
    # compute confusion matrix
    conf_mat = np.zeros(3) # TP, FN, FP

    # count occurences for every sample
    occur_true = np.sum(array_true==kw_token, axis=-1)
    occur_pred = np.sum(array_pred==kw_token, axis=-1)
    occur = np.column_stack([occur_true,occur_pred])

    # compute minimum between true and pred (offset, occurence in both)
    common_occur = np.min(occur,axis=-1)
    conf_mat[0] = np.sum(common_occur)
    # compute differences
    diff_occur = occur - common_occur.reshape(-1,1)
    conf_mat[1] = np.sum(diff_occur[:,0]) # more actual ones than predicted: FN
    conf_mat[2] = np.sum(diff_occur[:,1]) # mre predicted ones than actual: FP

    return conf_mat

def cm_from_raw_seqs(tokens_prob, true_tokens, num_kwd):

    # (a) categorical prediction: take the most probable token at each time instant
    # tensor of shape ( N, seq_len )
    tokens_pred = tf.argmax(tokens_prob,axis=-1)

    # (b) posterior handling 1: remove duplicates
    tokens_post = remove_serial_duplicates(tokens_pred)

    # (c) posterior handling 2: remove null token (of CTC)
    tokens_post_p1 = remove_specific_token(tokens_post, num_kwd+1)
    tokens_true_p1 = remove_specific_token(true_tokens.numpy(), num_kwd+1)

    # (d) posterior handling 3: remove [unk] token
    # (precision and recall require this)
    tokens_post_p2 = remove_specific_token(tokens_post_p1, 0)
    tokens_true_p2 = remove_specific_token(tokens_true_p1, 0)

    # (e) convert list of lists to numpy (to facilitate computation)
    tokens_post_p2n = list2numpy(tokens_post_p2)
    tokens_true_p2n = list2numpy(tokens_true_p2)

    # (f) compute confusion matrices
    confusion_matrix = np.zeros(( num_kwd, 3))
    for k in range(num_kwd):
        confusion_matrix[k,:] += cm_from_processed_seqs(tokens_true_p2n, tokens_post_p2n, k+1)

    return confusion_matrix


def export_performance(folder, keywords, conf_mat, precision, recall, f1):

    with open(os.path.join(folder,'performance.txt'), 'w') as f:
        f.write('Confusion Matrix\n')
        f.write('keyword     TP        FN        FP\n')
        for i in range(conf_mat.shape[0]):
            f.write('{:12s}{:10s}{:10s}{:s}\n'.format(
                keywords[i], str(int(conf_mat[i,0])), str(int(conf_mat[i,1])), str(int(conf_mat[i,2]))
            ))
        f.write('\nMetrics per Keyword\n')
        f.write('keyword     precision   recall      F1\n')
        for i in range(conf_mat.shape[0]):
            f.write('{:12s}{:12s}{:12s}{:s}\n'.format(
                keywords[i], '{:.4f}'.format(precision[i]), '{:.4f}'.format(recall[i]), '{:.4f}'.format(f1[i])
            ))
        f.write('\nAverage Metrics\n')
        f.write('Precision: {:.4f}\n'.format(np.mean(precision)))
        f.write('Recall   : {:.4f}\n'.format(np.mean(recall)))
        f.write('F1       : {:.4f}\n'.format(np.mean(f1)))


# ----------------------------------------------
#   O T H E R S
# ----------------------------------------------

# convert list of lists to numpy (padding with -1)
def list2numpy(listlist, padding=-1):
    max_len = max(map(len, listlist))
    numpy_data = np.array([ x + [padding]*(max_len-len(x)) for x in listlist ])
    return numpy_data



# ==================================================================