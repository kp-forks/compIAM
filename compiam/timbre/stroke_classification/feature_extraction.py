import os
import tqdm

import pandas as pd
import numpy as np
import essentia.standard as estd
from pathlib import Path

SPLIT_PARAMS = {
    "fs": 44100,
    "windowSize": 1024,
    "hopSize": 512,
    "NRG_threshold_ratio": 0.005
}
DESCRIPTORS_TO_DISREGARD = ['sfx', 'tristimulus', 'sccoeffs']

from sklearn import preprocessing
MIX_MAX_SCALER = preprocessing.MinMaxScaler()


def split_file(filename):
    """Function to define split boundaries based on a fixed energy threshold
    Args:
        filename (str): path to file to process
    """
    x = estd.MonoLoader(filename = filename, sampleRate = SPLIT_PARAMS.get("fs"))()
    NRG = []
    #Main windowing and feature extraction loop
    for frame in estd.FrameGenerator(
        x,
        frameSize = SPLIT_PARAMS.get("windowSize"),
        hopSize = SPLIT_PARAMS.get("hopSize"),
        startFromZero = True):
        NRG.append(estd.Energy()(frame))
    NRG = np.array(NRG)
    NRG = NRG / np.max(NRG)
    
    #Applying energy threshold to decide wave split boundaries
    split_decision_func = np.zeros_like(NRG)
    split_decision_func[NRG > SPLIT_PARAMS.get("NRG_threshold_ratio")] = 1
    #Setting segment boundaries
    #Inserting a zero at the beginning since we will decide the transitions using a diff function
    split_decision_func = np.insert(split_decision_func, 0, 0)
    diff_split_decision = np.diff(split_decision_func)
    #Start indexes: transition from 0 to 1
    start_indexes = np.nonzero(diff_split_decision > 0)[0] * SPLIT_PARAMS.get("hopSize")
    #Stop indexes: transition from 1 to 0
    stop_indexes = np.nonzero(diff_split_decision < 0)[0] * SPLIT_PARAMS.get("hopSize")
    return (x, NRG, split_decision_func, start_indexes, stop_indexes)

def process_strokes(stroke_dict, load_computed=False):
    """Process and extract features from stroke files
    Args:
        load_computed (bool): if True the pre-computed file is loaded
    """
    if not isinstance(load_computed, bool):
        raise ValueError("load_computed must be whether True or False") 
    first_one = True
    columns = []
    list_of_feat = []
    if load_computed == False:
        for stroke, files in tqdm.tqdm(stroke_dict.items()):
            for sample_file in files:
                #Get file id
                (x, _, _, start_indexes, stop_indexes) = split_file(sample_file)
                for start, stop in zip(start_indexes, stop_indexes):
                    x_seg = x[start: stop]
                    #Final check for amplitude (to avoid silent segments selection due to noise in split function)
                    if(np.max(np.abs(x_seg)) > 0.05):
                        #Amplitude normalisation
                        x_seg = x_seg / np.max(np.abs(x_seg))
                        #Compute and write features for file
                        features = estd.Extractor(
                            dynamics=False, rhythm=False, midLevel=False, highLevel=False)(x_seg)
                        feat = []
                        # Get descriptor names
                        descriptors = features.descriptorNames()
                        # Remove uneeded descriptors
                        for desc in DESCRIPTORS_TO_DISREGARD:
                            descriptors = [x for x in descriptors if desc not in x]
                        # Process MFCC
                        for i in np.arange(np.shape(features['lowLevel.mfcc'])[1]):
                            if first_one:
                                columns.append('mfcc' + str(i) + '.mean')
                                columns.append('mfcc' + str(i) + '.dev')
                            feat.append(np.mean(features['lowLevel.mfcc'][:, i]))
                            feat.append(np.std(features['lowLevel.mfcc'][:, i]))
                        # Now remove already computed mfcc
                        descriptors = [x for x in descriptors if 'mfcc' not in x]
                        for desc in descriptors:
                            if first_one:
                                columns.append(desc + '.mean')
                                columns.append(desc + '.dev')
                            feat.append(np.mean(features[desc]))
                            feat.append(np.std(features[desc]))
                        feat.append(stroke)
                        list_of_feat.append(feat)
                        if first_one:
                            columns = columns + ['stroke']
                            feature_list = columns
                            first_one = False
        # Convert list of features to dict and write to file
        df_features = pd.DataFrame(list_of_feat, columns=columns)
        df_features.to_csv(
            os.path.join(Path().absolute(), 'models', 'timbre', \
                'mridangam_stroke_classification', 'pre-computed_features.csv'), index=False)
    else: 
        # Load the pre-computed dict
        df_features = pd.read_csv(
            os.path.join(Path().absolute(), 'models', 'timbre', \
                'mridangam_stroke_classification', 'pre-computed_features.csv'))
        feature_list = list(df_features.columns)
    return df_features, feature_list

def normalize_features(trainig_data, feature_list=None):
    data_modif = trainig_data.copy()
    if feature_list is None:
        data_modif.iloc[:,:] =  MIX_MAX_SCALER.fit_transform(trainig_data.iloc[:,:].values)
    else: 
        data_modif.iloc[:,:len(feature_list)-1] = MIX_MAX_SCALER.fit_transform(trainig_data.iloc[:,:len(feature_list)-1].values)
    return data_modif

def features_for_pred(input_file):
    (audio, _, _, start_indexes, stop_indexes) = split_file(input_file)
    if len(start_indexes) > 1:
        max_len = np.argmax([np.abs(y - x) for x, y in zip(start_indexes, stop_indexes)])
    else:
        max_len = 0
    features = estd.Extractor(
        dynamics=False, rhythm=False, midLevel=False, highLevel=False)(
            audio[start_indexes[max_len]:stop_indexes[max_len]])   
    feat = []
    descriptors = features.descriptorNames()
    # Remove uneeded descriptors
    for desc in DESCRIPTORS_TO_DISREGARD:
        descriptors = [x for x in descriptors if desc not in x]
    # Process MFCC
    for i in np.arange(np.shape(features['lowLevel.mfcc'])[1]):
        feat.append(np.mean(features['lowLevel.mfcc'][:, i]))
        feat.append(np.std(features['lowLevel.mfcc'][:, i]))
    # Now remove already computed mfcc
    descriptors = [x for x in descriptors if 'mfcc' not in x]
    for desc in descriptors:
        feat.append(np.mean(features[desc]))
        feat.append(np.std(features[desc]))
    return feat