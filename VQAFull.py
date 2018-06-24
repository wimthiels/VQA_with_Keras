'''
VISUAL QUESTION ANSWERING : project Text Based Information Retrieval (part 2)

---------------------------------------------------------------------------------------------------------------------------------
## HOW TO RUN THIS PROGRAM ? >>>>>>>>>>>
==> all parameters can be modified using the CONFIG.txt file
==> but if you use the default parameters, make sure the following things are in place : 

1) these files must be in your home folder
    config file-> CONFIG.txt
    train file -> qa.894.raw.train.txt
    test file  -> qa.894.raw.test.txt
    pre trained word embeddings -> glove.6B.50d.txt
    image file -> img_features.csv

2) the kraino.utils folder must be somewhere in your python path. 
    IMPORTANT : the kraino folder delivered along with this script is made Python3 compatible.  so use this instead of the original

3) the wups score calculation requires the nltk.corpus package to be installed

4) the model architecture will be saved to a png file.  this requires graphviz.  You can always switch this off using(CREATE_GRAPHICAL_PLOT_MODEL = False)

==> Further remarks :
*) by default inference is done with and without the image for comparison (PREDICT_WITH_TEXTONLY=True ; PREDICT_WITH_IMAGE=True)
*) the model predictions + word indexes used are written to excel files (WRITE_EXCEL = True)
*) even when doing inference, the training data file needs to be in place (used to built word dictionaries)
*) the Keras warning ('No training configuration found in save file') when doing inference can be safely ignored
*) can be used interactively (INTERACTIVE_MODE = True), so you can ask questions via the command line (default = False)
---------------------------------------------------------------------------------------------------------------------------------

Created on 4 April 2018

tested with all latest versions 
-Keras 2.1.5
-Python 3.6.3
-Tensorflow 1.5.0
-Windows 10

@author: wim thiels
@article{malinowski2016tutorial,
  title={Tutorial on Answering Questions about Images with Deep Learning},
  author={Malinowski, Mateusz and Fritz, Mario},
  journal={arXiv preprint arXiv:1610.01076},
  year={2016}
  
}
'''

import re,os
from toolz import itemmap
import traceback
import logging
import pandas as pd

import numpy as np
from docutils.nodes import line
from numpy import asarray,random,zeros
from statistics import mean
np.random.seed(1337) # for reproducibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #it defaults to 0 (all logs shown), but can be set to 1 to filter out INFO logs, 2 to additionally filter out WARNING logs, and 3 to additionally filter out ERROR logs.


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,load_model
from keras.layers import Dense, Embedding, Input, LSTM, RepeatVector,concatenate,Dropout    
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping
from kraino.utils import print_metrics
from operator import itemgetter


def set_config_var():
    """
    Read in the configuration parameters used in execution (from CONFIG.TXT)
    """
    
    global FILE_TRAIN, FILE_TEST,FILE_PRE_TRAINED_EMBED,FILE_IMG_DATA,TRAINING_MODE,PREDICT_MODE, INTERACTIVE_MODE, TRAIN_TEXT_ONLY,TRAIN_VQA,SAVE_TEXT_ONLY,DEBUG_IC,WRITE_EXCEL
    global FULL_TEST_SET,NB_TEST_EXAMPLES_RANDOM,CREATE_GRAPHICAL_PLOT_MODEL,PREDICT_WITH_IMAGE,PREDICT_WITH_TEXTONLY,DIM_WORD_EMBED
    global MAX_ANSWER_TIME_STEPS,MAX_QUERY_TIME_STEPS, DIM_LATENT,DIM_HIDDEN_UNITS,BATCH_SIZE,EPOCHS_TEXT_ONLY,EPOCHS_VQA
    global INFER_AUTO_ENCODER,ALPHA_LOSS_WEIGHTS_TEXT_ONLY,ALPHA_LOSS_WEIGHTS_VQA
    global FIXED_TEST_EX
    
    d_config = {k:v.strip() for line in open('CONFIG.txt','r') for k,v in [line.split('=')]}
    
    # FILES
    FILE_TRAIN= d_config.get('FILE_TRAIN','qa.894.raw.train.txt')
    FILE_TEST= d_config.get('FILE_TEST','qa.894.raw.test.txt')
    FILE_PRE_TRAINED_EMBED= d_config.get('FILE_PRE_TRAINED_EMBED','glove.6B.50d.txt')
    FILE_IMG_DATA =  d_config.get('FILE_IMG_DATA','img_features.csv')
    
    # EXECUTION FLAGS
    TRAINING_MODE= str_to_bool(d_config.get('TRAINING_MODE','False'))           # Retrieves saved model from disk
    PREDICT_MODE= str_to_bool(d_config.get('PREDICT_MODE','False')) 
    INTERACTIVE_MODE = str_to_bool(d_config.get('INTERACTIVE_MODE','False'))     
    TRAIN_TEXT_ONLY= str_to_bool(d_config.get('TRAIN_TEXT_ONLY','False'))           # trains text only model, otherwise VQA model is trained
    TRAIN_VQA = str_to_bool(d_config.get('TRAIN_VQA','False'))     # trains vqa model
    SAVE_TEXT_ONLY = str_to_bool(d_config.get('SAVE_TEXT_ONLY','False'))          # save text only model, otherwise VQA model is trained
    DEBUG_IC =  str_to_bool(d_config.get('DEBUG_IC','False'))                # extra info for debugging
    WRITE_EXCEL =  str_to_bool(d_config.get('WRITE_EXCEL','True') )                # will output excels (question, predicted answer, true answer + extra columns useful for debugging)
    FULL_TEST_SET =  str_to_bool(d_config.get('FULL_TEST_SET','True'))            # if True, then the complete input file will be processed for testing
    NB_TEST_EXAMPLES_RANDOM =  int(d_config.get('NB_TEST_EXAMPLES_RANDOM',200 ))      # if FULL_TEST_SET = False, only this amount of samples will be selected (at random)
    CREATE_GRAPHICAL_PLOT_MODEL =  str_to_bool(d_config.get('CREATE_GRAPHICAL_PLOT_MODEL','True')) # creates .png of the models. graphviz must be installed
    PREDICT_WITH_IMAGE= str_to_bool(d_config.get('PREDICT_WITH_IMAGE','True'))             #predict answers using text and images
    PREDICT_WITH_TEXTONLY= str_to_bool(d_config.get('PREDICT_WITH_TEXTONLY','True') )         #predict answers using the text only 
    INFER_AUTO_ENCODER= str_to_bool(d_config.get('INFER_AUTO_ENCODER','True') )  
    FIXED_TEST_EX= str_to_bool(d_config.get('FIXED_TEST_EX','True') )   #use a fixed set of 50 examples from the test set
 
    # MODEL PARAMETERS
    MAX_ANSWER_TIME_STEPS=int(d_config.get('MAX_ANSWER_TIME_STEPS',2))
    MAX_QUERY_TIME_STEPS=int(d_config.get('MAX_QUERY_TIME_STEPS',15))
    DIM_LATENT=int(d_config.get('DIM_LATENT',512))
    DIM_HIDDEN_UNITS=int(d_config.get('DIM_HIDDEN_UNITS',512))
    BATCH_SIZE=int(d_config.get('BATCH_SIZE',32))
    EPOCHS_TEXT_ONLY=int(d_config.get('EPOCHS_TEXT_ONLY',10))
    EPOCHS_VQA=int(d_config.get('EPOCHS_VQA',10))
    ALPHA_LOSS_WEIGHTS_TEXT_ONLY = float(d_config.get('ALPHA_LOSS_WEIGHTS_TEXT_ONLY',0.8)) 
    ALPHA_LOSS_WEIGHTS_VQA = float(d_config.get('ALPHA_LOSS_WEIGHTS_TEXT_ONLY_VQA',0.8))         #gives more weight to the answer decoder vs the auto encoder during training
    
    # CONSTANTS
    DIM_WORD_EMBED=int(d_config.get('DIM_WORD_EMBED',50))
    
def str_to_bool(s):
    """
    helper function turning strings into booleans
    """
    if s == 'True':
        return True
    else:
        return False



def read_img_data():
    """
    Reads in the image data (pre encoded by CNN)
    Out:
        d_img_feat    : dictionary (key : image number (eg Image235), value : image vector (pre encoded by CNN)
        the number of image loaded
    """
    df_img_feat = pd.read_csv(FILE_IMG_DATA,index_col=0,header=None)
    d_img_feat = {k:v for (k,v) in zip(df_img_feat.index.tolist(),df_img_feat.values.tolist())}

    return d_img_feat, len(list(d_img_feat.values())[0])



def daquar_qa_raw(train_or_test='train'):
    """
    Parses the raw answers and question from the Daquar dataset
    In:
        train_or_test - indicate what file to use 'test' = FILE_TEST, or 'train' = FILE_TRAIN

    Out:
        dictionary of parsed data
        - x = questions
        - y = answers
        - img_name = image names eg 'image123'
        - img_idx = image index eg '123'
    """
    if train_or_test=='test':
        filepath=FILE_TEST
    else:
        filepath=FILE_TRAIN

    x_list = []
    y_list = []
    img_name_list = []
    img_ind_list = []  
    img_vec = [] 
    img_vec_empty = [0] * dim_img_feat   
    with open(filepath,'r') as f1:
        for i,line in enumerate(f1):
            if i%2==0:
                scan = re.search(r"(.*)(?:(?:on|in|of) (?:th|the|this|\s) image\d+\s*)(\?)",line)
                if not scan:                   
                    scan = re.search(r"(.*)(?:image\d+\s*)(\?)",line)
                if scan:
                    x_list.append("".join(scan.groups()).strip())
                    img_name = re.search(r"(image\d+)",line).group(1).strip()
                    img_name_list.append(img_name)
                    img_ind_list.append(re.search(r"(?:image)(\d+)",line).group(1))
                    img_vec.append(d_img_feat.get(img_name,img_vec_empty))
                else:
                    print("$$line missed", line)
                    x_list.append(line)
                    img_name_list.append("image0")
                    img_ind_list.append("0")
            else:
                y_list.append(line.strip())

    
    return {'x':x_list, 
            'y':y_list, 
            'img_name':img_name_list, 
            'img_ind': img_ind_list,
            'img_vec': img_vec
            }    


def tokenize_text(text,typeInput,maxlen,padding='post'):
    """
    Turns a raw text into a tokenized text (indexes instead of words) with an accompanying word to index dictionary.
    the resulting sentence is also padded to a given length
    and dictionary is expanded with some extra words (end, <pad>, unknown)
    In:
        text        - list of sentences
        typeInput   - indicate if the text is composed of 'questions' or 'answers'.  The terms added to word2index dictionary will be different
        maxlen      - output length of all the sentences (padding applied)
        padding     - type of padding, 'post' or 'pre'
    Out:
        wword_index    - dict mapping from word to index
        padded_text    - the tokenized and padded text
        encoded_text   - the tokenized text (no padding)
    """
    tok = Tokenizer()
    tok.fit_on_texts(text)
    
    # integer encode the queries
    encoded_text = tok.texts_to_sequences(text)  
     
    # pad queries to a max length with zeros
    padded_text= pad_sequences(encoded_text, maxlen=maxlen, padding=padding)
    
    # enrich the index with begin and end (we make the word_index zero indexed, by adding <pad> with zero index)
    word_index = tok.word_index
    max_value_idx = len(word_index)
    
    word_index['padding'] = 0
    if typeInput=='questions':
        word_index['unknown']= max_value_idx + 1
    if typeInput=='answers':
        word_index['end']= max_value_idx + 1
    
    
    # add end to mark end of sentence
    if padding=='post':
        for i,indexsentence in enumerate(padded_text):
            for j,indexword in enumerate(indexsentence):
                if indexword==0:
                    padded_text[i,j]=word_index['end']
                    break
            
    return word_index, padded_text, encoded_text

def tokenize_test_queries(raw_q,word2index,padding='pre'):
    """
    tokenizes new (test)queries with the index built from the training queries
    In:
        raw_q       - list of questions
        word2index  - index that will be used to tokenize the questions
        padding     - padding to be used, 'pre' or 'post'
    Out:
        padded_queries - tokenized and padded queries
    """
    queries = []
    for sentence in raw_q:
        query = []
        for word in sentence.split():
            wordindex = word2index.get(word)
            if not wordindex:
                wordindex = word2index['unknown'] 
            if word!='?':  #questionmark should not be recognized as an unknown word
                query.append(wordindex)
             
        queries.append(query)
    
    padded_queries= pad_sequences(queries, maxlen=MAX_QUERY_TIME_STEPS, padding=padding)    
    return padded_queries

def prep_answer_gen_input_data(padded_seqs, l_img_vec, word_index,max_nb_words):
    """
    prepares the decoder input data that will be used in training.  
    this is exactly the same as the tokenized training answers, but shifted 1 time position (<pad> at the first position)
    In:
        padded_seqs - list of answers (tokenized)
        l_img_vec   - list of image vectors
        word_index  - word to index for the answers of training set 
        max_nb_words- maximum number of words in the answer
    Out:
        answer_gen_input_data - input data for the decoder
        vis_input_data        - visual input for the answer generator (VQA-model)
    """
    answer_gen_input_data = np.zeros((len(padded_seqs), max_nb_words),dtype='float32')
    vis_input_data = np.zeros((len(padded_seqs),max_nb_words,dim_img_feat),dtype='float32')
    
    for i,seq in enumerate(padded_seqs):
        for j,index in enumerate(seq,1):
            if j==1:
                answer_gen_input_data[i,0] = word_index['padding']
            if j<max_nb_words:
                answer_gen_input_data[i,j] = index
            vis_input_data[i,j-1]=l_img_vec[i]
           
    return answer_gen_input_data,vis_input_data

def prep_auto_encoder_target_data(padded_seqs,embed_index):
    """
    Prepares the target data of the auto encoder used in training.  
    Return the a list of queries where every word in the query is in a 1-hot representation (dimension = vocab_size)
    In:
        padded_seqs - list of queries(tokenized)
        embed_index  - dict (key = word, value = word embedding)
    Out:
        auto_encoder_target_data - target data for the auto encoder (list of words where every word is a word embedding)
    """    
    auto_encoder_target_data = np.zeros((len(padded_seqs), MAX_QUERY_TIME_STEPS,DIM_WORD_EMBED),dtype='float32')
    for i,seq in enumerate(padded_seqs):
        for j,index in enumerate(seq):
            word = index2word_train_q[index]
            auto_encoder_target_data[i,j] = embed_index.get(word,np.zeros(DIM_WORD_EMBED))
   
    return auto_encoder_target_data

def prep_answer_gen_target_data(padded_seqs,max_nb_words,vocab_size):
    """
    Prepares the target data of the decoder used in training.  
    Return the a list of answers where every word in the answer is in a 1-hot representation (dimension = vocab_size)
    In:
        padded_seqs - list of answers (tokenized)
        max_nb_words- maximum number of words in the answer
        vocab_size  - number of different words in the answers of the training set
    Out:
        answer_gen_target_data - input data for the decoder
    """    
    answer_gen_target_data = np.zeros((len(padded_seqs), max_nb_words,vocab_size),dtype='float32')
    for i,seq in enumerate(padded_seqs):
        for j,index in enumerate(seq):
            answer_gen_target_data[i,j,index] = 1
   
    return answer_gen_target_data


    
def debug_check_repr_of_data(index,encoder_input_data_train,anwer_gen_input_data_train,answer_gen_target_data_train,train_raw_a,train_raw_q,index2word_train_a,index2word_train_q):
    """
    a print out of the important datastructures + some random checks.  for testing, debugging purposes
    """  
    print("index =" + str(index))
    print("RAW DATA------")
    print("train_raw_q =>" + train_raw_q[index])
    print("train_raw_a => " + train_raw_a[index])
    print("SHAPES-------")
    print("encoder_input_data shape =>" + str(encoder_input_data_train.shape))
    print("decoder_input_data shape =>" + str(anwer_gen_input_data_train.shape))
    print("decoder_target_data shape =>" + str(answer_gen_target_data_train.shape))
    print("decoder_target2_data shape =>" + str(auto_encoder_target_data_train.shape))

    print("INDEXES CHECK :the query, decoderinput reconstructed via the reverse index : ")
    print("->",str(" ".join([index2word_train_q[i] for i in encoder_input_data_train[index] if i!=0])))
    print("->",str(" ".join([index2word_train_a[i] for i in anwer_gen_input_data_train[index] if i!=0])))
    
    print("MODELINPUT-------")
    print("encoder_input_data =>" + str(encoder_input_data_train[index]))
    print("decoder_input_data => " + str(anwer_gen_input_data_train[index]))
    print("word2index answer dec first word => " + index2word_train_a[int(anwer_gen_input_data_train[index,0])])
    for i,bit in enumerate(answer_gen_target_data_train[index,0]):
        if bit==1:
            print("decoder_target_data first word : one hot=> " +  index2word_train_a[i])
    for i,bit in enumerate(answer_gen_target_data_train[index,1]):
        if bit==1:
            print("decoder_target_data second word : one hot=> " +  index2word_train_a[i])
    
    print("check auto_encoder_target_data_train")
    print("the last word of the encoder of this question = ", index2word_train_q[encoder_input_data_train[index,MAX_QUERY_TIME_STEPS-1]])
    print("the last word embedding of the target  = ", auto_encoder_target_data_train[index][MAX_QUERY_TIME_STEPS-1])
    print("this should match the word embeddingsindex for this word. see : ")     
    print(embed_index_q.get(index2word_train_q[encoder_input_data_train[index,MAX_QUERY_TIME_STEPS-1]])) 
    print("CHECK WORDEMBEDDING-------")
    print("check : the fifth word in the 4th question of training = chair. see..")
    i_1=padded_train_q[4,-1]
    w_1=index2word_train_q[i_1]
    print(w_1)
    print("the index of this word is ->", str(i_1)  )
    print("this word has the following representation(10char) in the word embed index .")
    print(str(embed_index_q[w_1][:10]))
    print("this should be the same as the weights in the embedding matrix...")
    print(str(embedding_matrix_q[i_1][:10]))
    print("--------answerembed-----------------")
    print("check : the first word in the 4th answer of training = red. see..")
    i_2=padded_train_a[4,0]
    w_2=index2word_train_a[i_2]
    print(w_2)
    print("the index of this word is ->", str(i_2)  )
    print("this word has the following representation(10char) in the word embed index .")
    print(str(embed_index_a[w_2][:10]))
    print("this should be the same as the weights in the embedding matrix...")
    print(str(embedding_matrix_a[i_2][:10]))
    print("-------------------------")
    print("CHECK QUESTION TOKENIZER-------")
    print("take this question with index ", index)
    print(train_raw_q[index])
    print("let's tokenize it and translate it back with the reverse index...")
    tok_query = tokenize_test_queries(train_raw_q[index:index+1], word2index_train_q, "pre").tolist()
    print(" ".join([index2word_train_q.get(int(i)) for i in tok_query[0]]))
    print("CHECK VISUAL INPUT ...")
    print("------------------------")
    image_name = train_img_name[index]
    print("this query --", train_raw_q[index], " -- asks a question about this image => ", image_name)
    print("this image has this representation in the imagefile (d_img_feat) (first 10 numbers : ")
    print(d_img_feat.get(image_name)[0:10])
    print("this should be the same as the representation i find in the image array at the index : ")      
    print(train_img_vec[index][0:10])      


def filterWordEmbedding(inputFilepath, s_filterWords , outputFilepath):
    """
    Filters out the pre-trained embedded word file (=big file).  Only the words used are kept and stored in a output file.  for performance reasons only
    In:
        inputFilepath - file with pre trained embeddings (glove)
        s_filterWords - list of words to keep for the input file
        outputFilepath- filtered version of the input file
    Out:
        keptWords - the words that are kept (not filtered)
    """  
    keptWords = []

    with open(inputFilepath,'r',encoding="utf8") as f1,open(outputFilepath,'w',encoding="utf8") as f2:
        for line in f1:
            word = line.split()[0]
            if word in s_filterWords:
                f2.write(line)
                keptWords.append(word)
    return keptWords


def create_pre_trained_embedding_matrix(f_pre_trained_embed,word2index,suffix):
    """
    Create the weight matrix for the embedding layer that uses pre-trained word embeddings
    In:
        f_pre_trained_embed - file with pre-trained word embeddings (glove)
        word2index          - mapping of a word to it's index
        suffix              - suffix to be used in filename

    Out:
        embedding_matrix    - weights for the embedding layer
        embeddings_index    - mapping of a word to it's word embedding
    """  
    #filter for performance reasons (big file...)
    f_pre_trained_embed_filter = f_pre_trained_embed.replace(".txt", ".filtered{}.txt".format(suffix))
    filterWordEmbedding(f_pre_trained_embed , word2index.keys(),f_pre_trained_embed_filter)
    
    # load the whole embedding into memory
    embeddings_index = dict()
    
    f = open(f_pre_trained_embed_filter,encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    if DEBUG_IC: print('Loaded %s word vectors.' % len(embeddings_index))
    
    no_rep_words = []
    # create a weight matrix for the different words 
    embedding_matrix = zeros((len(word2index)+1, 50))  
    for word, i in word2index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector 
        else:
            no_rep_words.append(word) #words that are not found will be represented with zeros
    
    if DEBUG_IC: print("$$no rep words => " + str(no_rep_words))
   
    return embedding_matrix,embeddings_index


def get_models_from_disk():
    """
    load a saved Keras model from disk
    """  
    
    try:
        print(">>>retrieving models from disk...")
        model_train_TextOnly= load_model('model_train_textOnly.h5')
        model_train_VQA = load_model('model_train_VQA.h5')
        
        model_encoder_text_only = load_model('model_encoder_text_only.h5')
        model_encoder_VQA = load_model('model_encoder_VQA.h5')
        model_encoder2 = load_model('model_encoder2.h5')

        model_answer_gen_textOnly = load_model('model_answer_gen_textOnly.h5')        
        model_answer_gen_VQA = load_model('model_answer_gen_VQA.h5')
        
        model_auto_encoder_textOnly = load_model('model_auto_encoder_textOnly.h5')
        model_auto_encoder_VQA = load_model('model_auto_encoder_VQA.h5')

    except Exception:
        logging.error(traceback.format_exc())
        return
    
    return [model_train_TextOnly,model_train_VQA,model_encoder_text_only, model_encoder_VQA,model_encoder2,model_answer_gen_textOnly,model_answer_gen_VQA,model_auto_encoder_textOnly,model_auto_encoder_VQA]
    

    
def predict_answers_to_queries(tok_q, l_img_vec=None):
    """
    predict answers to a list of queries using the inference model
    In:
        tok_q       - list of queries as input (tokenized)
        l_img_vec   - list of image vectors (if None then text only model will be used)
    Out:
        l_pred_anwers     - list of predicted answers (format = string)
        l_pred_answer_idx - list of predicted answers (format = word indexes)
        tok_q             - list of questions (format = word indexes)
        l_sent_hidden_rep - list of hidden representation of sentences (=hidden output state of the encoder)
    """  
    l_pred_anwers = []
    l_pred_answer_idx = []
    l_sent_hidden_rep = []
    
    for i in range(len(tok_q)):  
        if l_img_vec:
            pred_answer, pred_answer_idx, sent_hidden_rep = predict_query_answer(tok_q[i:i+1], l_img_vec[i:i+1])
        else:
            pred_answer, pred_answer_idx, sent_hidden_rep = predict_query_answer(tok_q[i:i+1])
            
        l_pred_anwers.append(pred_answer)
        l_pred_answer_idx.append(pred_answer_idx)
        l_sent_hidden_rep.append(sent_hidden_rep)

        
    return l_pred_anwers,l_pred_answer_idx,list(tok_q),l_sent_hidden_rep

def predict_query_answer(query,img_vec=None):
    """
    predict an answer to a query using the inference model
    In:
        query       - query as input (tokenized)
        img_vec     - image vector (if None then text only model will be used)
    Out:
        pred_answ        - predicted answer (format = string)
        pred_answer_idx  - predicted answer (format = word indexes)
    """    
    #initial states of decoder are the output states of encoder + image
    if img_vec:
        states_enc_q = model_encoder_VQA.predict(query)
    else:
        states_enc_q = model_encoder_text_only.predict(query)
        

    #initial iteration starts off with begin as input word
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2index_train_a['padding']
    
    #put imagevector in correct input format
    if img_vec:
        img_vec_input = np.zeros((1,1,dim_img_feat))
        img_vec_input[0,0] = np.asarray(img_vec)
    
    #go over every word until end is predicted or max length
    stop_condition = False
    pred_answ = ''
    pred_answer_idx = []
    while not stop_condition:
        if img_vec:
            output_tokens, h, c = model_answer_gen_VQA.predict([target_seq, img_vec_input]  + states_enc_q)
        else:
            output_tokens, h, c = model_answer_gen_textOnly.predict([target_seq]  + states_enc_q)

        # Sample a word
        sampled_word_index = np.argmax(output_tokens[0, -1, :]) #0-indexed off course (1 means second value in array)
        pred_answer_idx.append(sampled_word_index)

        sampled_word = index2word_train_a.get(sampled_word_index)
        if sampled_word != 'end' and sampled_word:
            if pred_answ == '':
                pred_answ += sampled_word
            else:
                pred_answ += ' ' + sampled_word

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == 'end' or
           len(pred_answ.split()) >= MAX_ANSWER_TIME_STEPS):
            stop_condition = True

        # Update the target sequence 
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_word_index

        # Update states
        states_enc_q = [h,c]


    return pred_answ, pred_answer_idx, states_enc_q[0]

def compute_cosine_similarity(x, y):
    """
    compute cosine similarity between 2 vectors
    In:
        x - first vector
        y - second vector
    Out:
        cosine similarity
    """  
    return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))  


def autoencode_queries(l_tok_pad_q, textOnly=False):
    """
    reconstruct a list of queries via the auto encoder model
    In:
        l_tok_pad_q              - list of tokenized (and padded) questions)
        textOnly                 - if True then the auto encoder from the text only model is used, otherwise the one from the VQA model
    Out:
        l_reconstructed_queries  - list of reconstructed queries (via the auto encoder model)
        reconstructed_embed_queries  - list of reconstructed queries where each word is a word embedding (so the direct output of the auto encoder)
    """   
    #run auto encoder
    if textOnly:
        reconstructed_embed_queries = model_auto_encoder_textOnly.predict(l_tok_pad_q).tolist()
    else:
        reconstructed_embed_queries = model_auto_encoder_VQA.predict(l_tok_pad_q).tolist()
    
    #pick as predicted word the word from the pre trained word embeddings that has the highest cosine similarity with the predicted word embedding
    l_l_reconstructed_queries = []
    for pred_sent in reconstructed_embed_queries:
        pred_word_sentence = []
        for pred_word_embed in pred_sent:
            cosim = [(k,compute_cosine_similarity(pred_word_embed, v)) for k,v in embed_index_q.items()]
            word_not_found = True
            while word_not_found:
                pred_word,v = max(cosim,key=itemgetter(1))
                if pred_word in pred_word_sentence:
                    if pred_word =='padding':
                        word_not_found = False
                    else:
                        cosim.remove((pred_word,v))
                else:
                    if pred_word !='padding':       
                        pred_word_sentence.append(pred_word)
                    word_not_found = False
        
        pred_word_sentence.append('?')   
        l_l_reconstructed_queries.append(pred_word_sentence)
    
    l_reconstructed_queries = [" ".join(word) for word in l_l_reconstructed_queries]
    
    return l_reconstructed_queries, reconstructed_embed_queries 
    
def write_prediction_vs_truth_excel(raw_q,raw_a,pred_a,l_reconstructed_q = None,pred_a_idx=None,
                                    padded_q=None,suffixFile="test",l_vis_rep=None,test_img_name_sel=None):
    """
    write an excel
    In:
        raw_q             - list of questions
        raw_a             - list of answers
        pred_a            - list of predicted answers
        pred_a_idx        - list of predicted answers in indexformat,
        padded_q          - list of questions (tokenized)
        l_reconstructed_q - list of reconstructed queries (via auto encoder)
        suffixFile        - suffix of excel file

    """    
    outputfilePath = "predictionAnwers.{}.csv".format(suffixFile)
    
    columns = ["question",
               "true answer",
               "predicted answer",
               "reconstructed query",
               "predicted answer(idx)",
               "question (idx)",
               "visual rep",
               "test_img_name_sel"
               ]
    
    if not l_reconstructed_q:l_reconstructed_q=[" "] * len(raw_q)
    
    df = pd.DataFrame({columns[0]:raw_q,
                       columns[1]:raw_a,
                       columns[2]:pred_a,
                       columns[4]:pred_a_idx,
                       columns[5]:padded_q,
                       columns[6]:l_vis_rep,
                       columns[7]:test_img_name_sel,
                       columns[3]:l_reconstructed_q})
    
    print(">>>writing excel file (", outputfilePath, ")...")
    df.to_csv(outputfilePath,columns=columns)
    
def write_index2word_excel(word2index,file_suffix='question'):
    """
    write an excel of a word index
    In:
        word2index             - word index
        file_suffix            - suffix of excel file
    """   
    outputfilePath = "index2word_{}.csv".format(file_suffix)
    
    df = pd.DataFrame.from_dict(word2index,orient='index')
    
    df.to_csv(outputfilePath)
    print("$$wrote excel => ", outputfilePath)
    

def get_WUPS_score(true_a, pred_a):
    """
    evaluate the model (wups and accuracy)
    In:
        true_a  - ground truth
        pred_a  - predictions
    Out:
        wups    - wups score
    """    
    wups = print_metrics.select['wups'](
        gt_list=true_a,
        pred_list=pred_a,
        verbose=0,
        extra_vars=None)
    
    return wups

def eval_sent_rep_auto_encoder(l_sent_l_word_embed):
    """
    evaluate the auto encoder by looking at the latent representation of the queries (= output of the encoder)
    The latent representation of the original query is compared to the latent representation of the reconstructed query. (cosine similarity)
    This is done by feeding the reconstructed query back into the encoder
    In:
        l_sent_l_word_embed  - the reconstructed queries via the auto encoder.  words are represented as word embeddings(DIM_WORD_EMBED)
    Out:
        cossim_sent     - a list of cosine similarities per sentence
        avg_cossim_sent -  the average of cossim_sent = the overall evaluation of the auto encoder
    """   

    l_sent_test = [[embedding_matrix_q[wordindex] for wordindex in sentence] for sentence in l_tok_pad_test_q_sel]
    
    l_sent_hidden_rep = model_encoder2.predict(asarray(l_sent_test))[0]
    l_sent_hidden_rep_comp = model_encoder2.predict(asarray(l_sent_l_word_embed))[0]
    
    cossim_sent = [float(compute_cosine_similarity(x, y)) for (x,y) in zip(l_sent_hidden_rep,l_sent_hidden_rep_comp)]
    avg_cossim_sent = mean(cossim_sent)
    
    return cossim_sent, avg_cossim_sent


def calc_acc_auto_encoder(original_queries, reconstructed_queries):
    """
    to evaluate the accuracy of the auto encoder we check how many words in the reconstructed query match those in the raw query.
    so 3 words correct out of 15 will give an accuracy of 3/15 for that sentence.  
    total accuracy =  (accuracy summed up over all sentences / number of sentences) * 100
    
    remarks: 
        * This can only be sensibly used if the reconstructed query is not allowed to have repeating words.
        * not without flaws => this ignores word order, favors short answers..
    
    In:
        original_queries       - ground truth
        reconstructed queries  - predictions
    Out:
        acc_total              - accuracy of the auto encoder
        l_acc_sent             - list of accuracies, per sentence
    """    
    l_acc_sent = [] 
    for idx_sent, raw_sent in enumerate(original_queries):  
        raw_sent_split = raw_sent.split()
        del raw_sent_split [-1] #get rid of '?'
        word_hits = [word in raw_sent_split for word in reconstructed_queries[idx_sent].split()]
        acc_sent = sum(word_hits) /  (len(word_hits) - 1 ) #'?' adds 1 to the length, correct this 
        l_acc_sent.append(acc_sent)

    acc_total = (sum(l_acc_sent)/len(original_queries))*100

    return acc_total, l_acc_sent

def debug_run_inference_with_training_data(use_image=True):    
    """
    in debug mode we also check the performance on the training data (check overfitting, compare with test results..)
    """
    #B064-do the same inference with the training data to check results           
    print("starting inference with trainingdata....")
    sample_set_of_3 = [4,6,8]
        
    indices_to_see = random.randint(low=0, high=len(train_raw_q), size=NB_TEST_EXAMPLES_RANDOM)
    indices_to_see_2 = np.concatenate((np.asarray(sample_set_of_3),indices_to_see),axis=0)
    train_raw_a_sel = [train_raw_a[i] for i in indices_to_see_2]
    train_raw_q_sel = [train_raw_q[i] for i in indices_to_see_2]
    train_vis_sel = [train_img_vec[i] for i in indices_to_see_2]

    l_tok_pad_train_q_sel = tokenize_test_queries(train_raw_q_sel,word2index_train_q,"pre")
    
    if use_image:    
        l_pred_anwers_train, l_pred_answer_idx_train, l_padded_queries_train,_ = predict_answers_to_queries(l_tok_pad_train_q_sel,train_vis_sel)
    else:
        l_pred_anwers_train, l_pred_answer_idx_train, l_padded_queries_train,_ = predict_answers_to_queries(l_tok_pad_train_q_sel)
            
    l_reconstructed_queries_train,_ = autoencode_queries(l_tok_pad_train_q_sel,not use_image)
    
    if use_image:
        suff_excel = "train_VQA"
    else:
        suff_excel = "train_textOnly"
        
    if WRITE_EXCEL:
        write_prediction_vs_truth_excel(train_raw_q_sel,train_raw_a_sel,l_pred_anwers_train,l_reconstructed_queries_train,
                    l_pred_answer_idx_train,
                    l_padded_queries_train,suff_excel,train_vis_sel,None)
        write_index2word_excel(word2index_train_q,"question")
        write_index2word_excel(word2index_train_a,"answer")
    print("...end inference with trainingdata.  use_image = ", use_image)
    
    print("WUPS for training : ")
    print("Tested on ", len(train_raw_q_sel), " examples :")
    l_gt_anwers_train = [answ.replace('_', ' ') for answ in train_raw_a_sel]
    l_gt_anwers_train = [answ.replace(',', ' ') for answ in l_gt_anwers_train]
    wups_score_train = get_WUPS_score(l_gt_anwers_train,l_pred_anwers_train)
    print("----> accuracy for the answer generator (train) = ",  '{0:.2f}%'.format(wups_score_train[0].get("value")))
    print("----> WUPS@0.9 for the answer generator (train) = ",  '{0:.2f}%'.format(wups_score_train[1].get("value")))
    print("----> accuracy for the auto encoder = ", '{0:.2f}%'.format(calc_acc_auto_encoder(train_raw_q_sel, l_reconstructed_queries_train)[0]))


def evaluate_reconstructed_queries():     
    """
    evaluate the auto encoder 
    """
#    wups_score_acc = get_WUPS_score(test_raw_q_sel,l_reconstructed_queries_test)
#    print("--> accuracy for the auto encoder = ", wups_score_acc[0].get("value"))  # normal accuracy will always be zero = too strict
    print("auto encoder >>") 
    print("--> accuracy (soft version) = ", '{0:.2f}%'.format(calc_acc_auto_encoder(test_raw_q_sel, l_reconstructed_queries_test)[0]))
    _,avg_cossim = eval_sent_rep_auto_encoder(reconstructed_embed_queries_test)    
    print("--> average cosine similarity between the latent query representation (original query vs reconstructed query) = ", '{0:.2f}'.format(avg_cossim))
   

def evaluate_predicted_answers(use_image=True):
    """
    evaluate the predicted answers for the 2 models and print out results
    
    In :
        use_image    : if True , the data comes from the VQA model, otherwise the text only model
    
    """
    print("-----------------------------")
    if use_image:
        print("EVALUATE MODEL (TEXT + IMAGE):")
    else:
        print("EVALUATE MODEL (TEXT ONLY):")
    print("-----------------------------")
    
    l_gt_anwers_test = [answ.replace('_', ' ') for answ in test_raw_a_sel]
    l_gt_anwers_test = [answ.replace(',', ' ') for answ in l_gt_anwers_test]
    wups_score_test = get_WUPS_score(l_gt_anwers_test,l_pred_anwers_test)
#    print("Tested on ", len(test_raw_q_sel), " examples.  Trained on ", len(train_raw_q), " examples, ", "max epochs = ", EPOCHS, " (early stopping applied) : ")
    print("Tested on ", len(test_raw_q_sel), " examples")
    print("answer generator >>")
    print("--> accuracy  = ",  '{0:.2f}%'.format(wups_score_test[0].get("value")))
    print("--> wups at 0.9 = ",  '{0:.2f}%'.format(wups_score_test[1].get("value")))
    
    
if __name__ == '__main__':
    #A005-INITIALIZATIONS----------------------------------------------------------
    print(">>>reading in data...")
    set_config_var()
    
    #A010-READ DATA----------------------------------------------------------
    #--B010-read in data from file
    d_img_feat, dim_img_feat = read_img_data()
    
    train_data_representation  = daquar_qa_raw('train')
    train_raw_q = train_data_representation['x']  #retrieve the train questions 
    train_raw_a = train_data_representation['y']  #retrieve the train answers
    train_img_vec = train_data_representation['img_vec']  #retrieve the visual representations
    train_img_name = train_data_representation['img_name']  #retrieve the image names
    
    test_data_representation  = daquar_qa_raw('test')
    test_raw_q = test_data_representation['x']  #retrieve the test questions 
    test_raw_a = test_data_representation['y']  #retrieve the test answers
    test_vis = test_data_representation['img_vec']  #retrieve the visual representations
    test_img_name = test_data_representation['img_name']  #retrieve the image names
    
    
    #--B011-some stats
    max_nb_words_train_q =  max([len(x.split()) for x in train_raw_q]) + 2
    max_nb_words_train_a =  max([len(x.split()) for x in train_raw_a]) + 2

    #A020-TEXT-PREPROCESSING----------------------------------------------------------
        #--B020-Tokenize-queries and answers(train)
    print(">>>preprocessing the data...")
    word2index_train_q, padded_train_q, enc_train_q = tokenize_text(train_raw_q,'questions',MAX_QUERY_TIME_STEPS,'pre')
    word2index_train_a, padded_train_a, enc_train_a = tokenize_text(train_raw_a,'answers',MAX_ANSWER_TIME_STEPS,'post')
    vocab_size_train_q = len(word2index_train_q)
    vocab_size_train_a = len(word2index_train_a)

        #--B021-build reverse indexes 
    index2word_train_q = itemmap(reversed, word2index_train_q)  
    index2word_train_a = itemmap(reversed, word2index_train_a)  

    
        #--B022-prepare pretrained word embeddings (fixed weights)
    embedding_matrix_q,embed_index_q = create_pre_trained_embedding_matrix(FILE_PRE_TRAINED_EMBED,word2index_train_q,".Queries")
    embedding_matrix_a,embed_index_a = create_pre_trained_embedding_matrix(FILE_PRE_TRAINED_EMBED,word2index_train_a,".Answers")

        #--B023-prepare model data(inputs and target)
    encoder_input_data_train = padded_train_q
    anwer_gen_input_data_train, vis_input_data_train = prep_answer_gen_input_data(padded_train_a, train_img_vec, word2index_train_a,MAX_ANSWER_TIME_STEPS)
    answer_gen_target_data_train = prep_answer_gen_target_data(padded_train_a, MAX_ANSWER_TIME_STEPS, vocab_size_train_a)
    auto_encoder_target_data_train = prep_auto_encoder_target_data(padded_train_q,embed_index_q)
    
    if DEBUG_IC:
        debug_check_repr_of_data(4,encoder_input_data_train,anwer_gen_input_data_train,answer_gen_target_data_train,train_raw_a,train_raw_q,index2word_train_a,index2word_train_q)
    
    #A030-BUILD MODELS----------------------------------------------------------    
    if not TRAINING_MODE:
        models = get_models_from_disk()
        if models:
            model_train_TextOnly,model_train_VQA,model_encoder_text_only,model_encoder_VQA,model_encoder2,model_answer_gen_textOnly,model_answer_gen_VQA,model_auto_encoder_textOnly,model_auto_encoder_VQA = models
    else:          
        #B030-DEFINE MODEL ARCHITECTURE----------------------------------------------------------
        print("Building model architecture ..")
            #--C030-define encoder
        encoder_inputs = Input(shape=(None,),name='Encoder_input_query')
        e1 = Embedding(vocab_size_train_q+1, DIM_WORD_EMBED, weights=[embedding_matrix_q], trainable=False,name='encoder_word_embed')(encoder_inputs)
        encoder2_inputs = Input(shape=(None,DIM_WORD_EMBED),name='Encoder2_input_query')
        encoder_lstm = LSTM(DIM_LATENT,return_state=True,name='encoder_lstm')
        x, state_h, state_c = encoder_lstm(e1)
        x2, state2_h, state2_c = encoder_lstm(encoder2_inputs)
        encoder_states = [state_h, state_c]
        encoder_states2 = [state2_h, state2_c]
        
               
            #--C031-define the answer generator
        answer_generator_inputs = Input(shape=(None,),name='answer_gen_input')
        answer_gen_embed = Embedding(vocab_size_train_a + 1, DIM_WORD_EMBED, weights=[embedding_matrix_a], 
                                     trainable=False,name='answer_gen_embedding')(answer_generator_inputs)

        answer_gen_lstm = LSTM(DIM_LATENT, return_sequences=True, return_state=True,name='answer_gen_lstm')
        answer_gen_lstm_outputs, _, _ = answer_gen_lstm(answer_gen_embed,initial_state=encoder_states)
        
            #--C032-define classifier (text only)      
        answer_gen_dense_textOnly = Dense(vocab_size_train_a, activation='softmax',name='answer_gen_softmax_textOnly')
        answer_gen_outputs_text = answer_gen_dense_textOnly(answer_gen_lstm_outputs)          

        
            #--C033-define classifier (text + image)
        visual_input = Input(shape=(None,dim_img_feat),name='visual_input')
        conc_vis_answer  = concatenate([answer_gen_lstm_outputs, visual_input],axis=2,name='answer_image_concat') 
        
        answer_gen_hidden1 = Dense(DIM_HIDDEN_UNITS, kernel_initializer="uniform",activation='tanh',name='answer_gen_hidden1')
        answer_gen_LSTM_outputs = answer_gen_hidden1(conc_vis_answer)
        
        answer_gen_dropout1 = Dropout(0.5)
        answer_gen_LSTM_outputs = answer_gen_dropout1 (answer_gen_LSTM_outputs)

        answer_gen_hidden2 = Dense(DIM_HIDDEN_UNITS, kernel_initializer="uniform",activation='tanh',name='answer_gen_hidden2')
        answer_gen_LSTM_outputs = answer_gen_hidden2(answer_gen_LSTM_outputs)
        
        answer_gen_dropout2 = Dropout(0.5)
        answer_gen_LSTM_outputs = answer_gen_dropout2 (answer_gen_LSTM_outputs)
        
        answer_gen_dense_VQA = Dense(vocab_size_train_a, activation='softmax',name='answer_gen_softmax_VQA')
        answer_gen_outputs_VQA = answer_gen_dense_VQA(answer_gen_LSTM_outputs)
        
        
            #--C034-define repeat_encoded_question (the autoencoder)
        repeat_encoded_question = RepeatVector(MAX_QUERY_TIME_STEPS,name='repeater_autoencoder')(state_h)
        auto_encoder_outputs = LSTM(DIM_WORD_EMBED,return_sequences=True,name='auto_encoder_lstm')(repeat_encoded_question)
             
            
        if TRAIN_TEXT_ONLY:
        #B040-RUN TRAINING TextOnly model--------------------
            print(">>>run training Text only model...")
            model_train_TextOnly = Model([encoder_inputs ,answer_generator_inputs], 
                                  [answer_gen_outputs_text,auto_encoder_outputs])
            if CREATE_GRAPHICAL_PLOT_MODEL:
                plot_model(model_train_TextOnly, to_file='model_train_TextOnly.png', show_shapes=True)
                print(model_train_TextOnly.summary())  
                
                
            model_train_TextOnly.compile(optimizer='rmsprop', 
                              loss=['categorical_crossentropy','mse'],
                              loss_weights=[ALPHA_LOSS_WEIGHTS_TEXT_ONLY,1-ALPHA_LOSS_WEIGHTS_TEXT_ONLY])

          
            model_train_TextOnly.fit([encoder_input_data_train, anwer_gen_input_data_train], 
                          [answer_gen_target_data_train,auto_encoder_target_data_train],
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS_TEXT_ONLY,
                          validation_split=0.2)
                        #callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=1)])    

            if SAVE_TEXT_ONLY:
                model_train_TextOnly.save('model_train_TextOnly.h5',overwrite=True)
            
    
            #B050-BUILD INFERENCEMODELS TEXT ONLY----------------------------------------------------------
            print(">>>Building inference models text only...")
                #C050-build encoder model
            model_encoder_text_only = Model(encoder_inputs, encoder_states)
            if SAVE_TEXT_ONLY:
                model_encoder_text_only.save('model_encoder_text_only.h5',overwrite=True)
        
            if CREATE_GRAPHICAL_PLOT_MODEL:
                plot_model(model_encoder_text_only, to_file='model_encoder_text_only.png', show_shapes=True)
            
                     
                #C051-build answer generator model (common part = LSTM)
            answer_gen_state_input_h = Input(shape=(DIM_LATENT,),name='answer_gen_state_input_h')
            answer_gen_state_input_c = Input(shape=(DIM_LATENT,),name='answer_gen_state_input_c')
            answer_gen_states_inputs = [answer_gen_state_input_h, answer_gen_state_input_c]
        
            answer_gen_LSTM_outputs, state_h, state_c = answer_gen_lstm(
                answer_gen_embed, initial_state=answer_gen_states_inputs)
            answer_gen_states = [state_h, state_c]

                #C052-build answer generator model (Text only part)        
            answer_gen_outputs_textOnly = answer_gen_dense_textOnly(answer_gen_LSTM_outputs)
        
            model_answer_gen_textOnly = Model(
                [answer_generator_inputs] + answer_gen_states_inputs,
                [answer_gen_outputs_textOnly] + answer_gen_states)
            
            if SAVE_TEXT_ONLY:
                model_answer_gen_textOnly.save('model_answer_gen_textOnly.h5',overwrite=True)
            if CREATE_GRAPHICAL_PLOT_MODEL:
                plot_model(model_answer_gen_textOnly, to_file='model_answer_gen_textOnly.png', show_shapes=True)        
         
                #C053-build auto_encoder model (Text only part)          
            model_auto_encoder_textOnly = Model(encoder_inputs, auto_encoder_outputs)
            if SAVE_TEXT_ONLY:
                model_auto_encoder_textOnly.save('model_auto_encoder_textOnly.h5',overwrite=True)

                #C054 build alternate encoder (a new encoder model is built that directly start from word embeddings as input (as opposed to word indexes) (used in evaluation))
            model_encoder2 = Model(encoder2_inputs,encoder_states2)
            if SAVE_TEXT_ONLY: 
                model_encoder2.save('model_encoder2.h5',overwrite=True)
        
            print('Text_only models have been trained')
            if SAVE_TEXT_ONLY:
                print('and saved to disk')
        if TRAIN_VQA:
            #B060-RUN TRAINING VQA model--------------------
            print("run training VQA model...")    
            model_train_VQA = Model([encoder_inputs, visual_input ,answer_generator_inputs], 
                                  [answer_gen_outputs_VQA,auto_encoder_outputs])
            if CREATE_GRAPHICAL_PLOT_MODEL:
                plot_model(model_train_VQA, to_file='model_train_VQA.png', show_shapes=True)
                print(model_train_VQA.summary())
   
             
        
            model_train_VQA.compile(optimizer='rmsprop', 
                              loss=['categorical_crossentropy','mse'],
                              loss_weights=[ALPHA_LOSS_WEIGHTS_VQA,1-ALPHA_LOSS_WEIGHTS_VQA])

          
            model_train_VQA.fit([encoder_input_data_train, vis_input_data_train, anwer_gen_input_data_train], 
                          [answer_gen_target_data_train,auto_encoder_target_data_train],
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS_VQA,
                          validation_split=0.2)
                        # callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=1)])
    
            model_train_VQA.save('model_train_VQA.h5',overwrite=True)
                 
     
            #B070-BUILD INFERENCEMODELS VQA----------------------------------------------------------     
            print(">>>Building inference models VQA...")
                #C070-build encoder model VQA
            model_encoder_VQA = Model(encoder_inputs, encoder_states)
            model_encoder_VQA.save('model_encoder_VQA.h5',overwrite=True)
        
            if CREATE_GRAPHICAL_PLOT_MODEL:
                plot_model(model_encoder_VQA, to_file='model_encoder_VQA.png', show_shapes=True) 

                #C071-build answer generator model (common part = LSTM)
            answer_gen_state_input_h = Input(shape=(DIM_LATENT,),name='answer_gen_state_input_h')
            answer_gen_state_input_c = Input(shape=(DIM_LATENT,),name='answer_gen_state_input_c')
            answer_gen_states_inputs = [answer_gen_state_input_h, answer_gen_state_input_c]
        
            answer_gen_LSTM_outputs, state_h, state_c = answer_gen_lstm(
                answer_gen_embed, initial_state=answer_gen_states_inputs)
            answer_gen_states = [state_h, state_c]
       
                #C072-build answer generator model (VQA part)        
            conc_vis_answer  = concatenate([answer_gen_LSTM_outputs, visual_input],axis=2,name='answer_image_concat')        
            answer_gen_output = answer_gen_hidden1(conc_vis_answer)
            answer_gen_output = answer_gen_hidden2(answer_gen_output)
            answer_gen_outputs_VQA = answer_gen_dense_VQA(answer_gen_output)
        
            model_answer_gen_VQA = Model(
                [answer_generator_inputs,visual_input] + answer_gen_states_inputs,
                [answer_gen_outputs_VQA] + answer_gen_states)
        
            model_answer_gen_VQA.save('model_answer_gen_VQA.h5',overwrite=True)
            if CREATE_GRAPHICAL_PLOT_MODEL:
                plot_model(model_answer_gen_VQA, to_file='model_answer_gen_VQA.png', show_shapes=True)
            
            
                #C072-build auto encoder model VQA
            model_auto_encoder_VQA = Model(encoder_inputs, auto_encoder_outputs)
            model_auto_encoder_VQA.save('model_auto_encoder_VQA.h5',overwrite=True)
            if CREATE_GRAPHICAL_PLOT_MODEL:
                plot_model(model_auto_encoder_VQA, to_file='model_auto_encoder_VQA.png', show_shapes=True)   
            
            print('VQA models have been trained and saved to disk')

    #A060-RUN TESTDATA----------------------------------------------------------
    if PREDICT_MODE:
    
        #B060-take full test set or only a sample set of random questions    
        if FULL_TEST_SET:
            test_raw_a_sel = test_raw_a
            test_raw_q_sel = test_raw_q
            test_vis_sel = test_vis
            test_img_name_sel = test_img_name
        else:
            if FIXED_TEST_EX:
                indices_to_see = [1882, 3886, 5645, 1901, 2186, 2073, 1902, 3085,  219, 5035,  354 ,1652 ,  24 ,1084, 1135,3824 ,4286, 5296 ,  33 , 964,3873, 1018, 3865 ,3610 ,1963, 4394 ,3077, 1325,  605,  708, 1148 ,1530, 2016 , 886,  795,  426,  365, 4452, 3853, 3724, 4835, 5610, 1900, 5455, 3480 ,4137 ,5454,   72 ,3683]
            else:           
                indices_to_see = random.randint(low=0, high=len(test_raw_q), size=NB_TEST_EXAMPLES_RANDOM)
            test_raw_a_sel = [test_raw_a[i] for i in indices_to_see]
            test_raw_q_sel = [test_raw_q[i] for i in indices_to_see]
            test_vis_sel = [test_vis[i] for i in indices_to_see]
            test_img_name_sel =  [test_img_name[i] for i in indices_to_see]
        
            #B061-tokenize the test queries (+ padding)
        l_tok_pad_test_q_sel = tokenize_test_queries(test_raw_q_sel,word2index_train_q,"pre")
    
            #B062-reconstruct queries via the auto encoder
        if INFER_AUTO_ENCODER:
            print("\n>>> RECONSTRUCTING QUERIES...")
            l_reconstructed_queries_test, reconstructed_embed_queries_test  = autoencode_queries(l_tok_pad_test_q_sel)  
            evaluate_reconstructed_queries()
        else:
            l_reconstructed_queries_test = []
            reconstructed_embed_queries_test = []
                   
            #B063-predict answers from tokenized queries and evaluate
        if PREDICT_WITH_TEXTONLY:
            print("\n>>> PREDICTING ANSWERS (text only)...")
            l_pred_anwers_test, l_pred_answer_idx_test, l_padded_queries_test, l_sent_hidden_rep_test = predict_answers_to_queries(l_tok_pad_test_q_sel)
         
            if DEBUG_IC:  
                debug_run_inference_with_training_data(use_image=False)
                
            evaluate_predicted_answers(use_image=False)   
               
            if WRITE_EXCEL:          
                write_prediction_vs_truth_excel(test_raw_q_sel,test_raw_a_sel,l_pred_anwers_test,l_reconstructed_queries_test,l_pred_answer_idx_test,
                        l_padded_queries_test,"test_textOnly",None, test_img_name_sel)     
       
        if PREDICT_WITH_IMAGE:
            print("\n>>> PREDICTING ANSWERS(text + image)...")
            l_pred_anwers_test, l_pred_answer_idx_test, l_padded_queries_test, l_sent_hidden_rep_test = predict_answers_to_queries(l_tok_pad_test_q_sel, test_vis_sel)
        
            if DEBUG_IC:  
                debug_run_inference_with_training_data(use_image=True)
            evaluate_predicted_answers(use_image=True)

            if WRITE_EXCEL:  
                write_prediction_vs_truth_excel(test_raw_q_sel,test_raw_a_sel,l_pred_anwers_test,l_reconstructed_queries_test,l_pred_answer_idx_test,
                        l_padded_queries_test,"test_VQA",test_vis_sel,test_img_name_sel)    
    
    #A070-INTERACTIVE QUESTIONS VIA COMMANDSHELL---------------------------------------------------------
    if INTERACTIVE_MODE:
        print(">>> GOING INTERACTIVE...")
        while True:
            reply = input('Enter your query (stop to exit):')
            if reply == 'stop': break
            test_raw_q_sel=[]
            scan = re.search(r"(.*)(?:(?:on|in|of) (?:th|the|this|\s) image\d+\s*)(\?)",reply)
            if not scan:                   
                scan = re.search(r"(.*)(?:image\d+\s*)(\?)",reply)
            if scan:
                test_raw_q_sel.append("".join(scan.groups()).strip())
                img_name = re.search(r"(image\d+)",reply).group(1).strip()
            
            if not scan:
                print('<invalid question, try again>')  
                continue  
            
            test_vis_sel=d_img_feat.get(img_name,None)
            if not test_vis_sel:
                print('<image not found in database , try again...>')
                continue
    
            
            l_tok_pad_test_q_sel = tokenize_test_queries(test_raw_q_sel,word2index_train_q,"pre")
            l_pred_anwers_test, l_pred_answer_idx_test, l_padded_queries_test, l_sent_hidden_rep_test = predict_answers_to_queries(l_tok_pad_test_q_sel, test_vis_sel)
            print('>>>'," ".join(l_pred_anwers_test))
            
            
    
    
    


        
