# Development of a Voice Activity Detection System for DIRHA, based on GMMs and HLDA
ROOTDIR=/rmt/work/speech_activity_detection
SCRIPTSDIR=$ROOTDIR/scripts/PLP_VAD/python_src/plp_vad_gmm

# First add the various python modules to the PYTHONPATH
export PYTHONPATH=/rmt/work/speech_activity_detection/scripts/PLP_VAD/python_src:/rmt/work/speech_activity_detection/resources/Onyx-1.0.511/py

# Prepare audio files, transcriptions as well as lists of files
#./prepare.sh

# Configuration file
CONFIG=config/dirha_hlda.cfg

# Run experiments using configuration file
python $SCRIPTSDIR/dirha_experiments.py -c $CONFIG --working_dir $ROOTDIR/experiments/dirha_gmm_mfcc

# Run experiments using configuration file and specify certain parameters from command line
python $SCRIPTSDIR/dirha_experiments.py -c $CONFIG --working_dir $ROOTDIR/experiments/dirha_hlda_mfcc --acc_frames 11 --apply_hlda on\
    --feature_type MFCC_0_Z --n_features 13 --hlda_nuisance_dims 104



