mkdir -p audio
mkdir -p lab
lab_dir=/rmt/work/speech_activity_detection/Transcriptions
training_script=lists/train.script
testing_script=lists/test.script

rm -f $training_script $testing_script
for i in `ls /rmt5/databases/DIRHA/DIRHA_simcorpora2_v3/GR/dev1/simulations/sim*/Signals/Mixed_Sources/*/*/*.wav | sed /beam/d`; 
do 
    sim_id=`echo $i | grep -o 'sim[[:digit:]]\+'`;
    mic_id=`basename $i .wav`;
    new_id=dev1_${sim_id}_${mic_id}
    new_audio_file_name=audio/${new_id}.wav
    new_lab_file_name=lab/${new_id}.lab
    
    ln -s $i $new_audio_file_name;
    ln -s $lab_dir/${mic_id}/${mic_id}`echo $sim_id | sed s/sim//`.lab $new_lab_file_name; 
    echo $new_audio_file_name $new_lab_file_name >> $training_script;
done;

lab_dir=/rmt/work/speech_activity_detection/Transcriptions_test1
for i in `ls /rmt5/databases/DIRHA/DIRHA_simcorpora2_v3/GR/test1/simulations/sim*/Signals/Mixed_Sources/*/*/*.wav | sed /beam/d`; 
do 
    sim_id=`echo $i | grep -o 'sim[[:digit:]]\+'`;
    mic_id=`basename $i .wav`;
    new_id=test1_${sim_id}_${mic_id}
    new_audio_file_name=audio/${new_id}.wav
    new_lab_file_name=lab/${new_id}.lab
    ln -s $i $new_audio_file_name;
    ln -s $lab_dir/${mic_id}/${mic_id}`echo $sim_id | sed s/sim//`.lab $new_lab_file_name; 
    echo $new_audio_file_name $new_lab_file_name >> $testing_script;
done;



