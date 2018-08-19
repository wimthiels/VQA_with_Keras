# VQA_with_Keras
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
