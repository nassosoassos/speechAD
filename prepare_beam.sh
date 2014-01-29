#mkdir -p audio
#mkdir -p lab
lab_dir=/rmt/work/speech_activity_detection_git/lab/transcr_beam/dev1
training_script=lists/train.script_beam
testing_script=lists/test.script_beam

#rm -f $training_script $testing_script
for i in `ls /rmt5/databases/DIRHA/DIRHA_simcorpora2_v3/GR/dev1/simulations/sim*/Signals/Mixed_Sources/*/*/*.wav | grep "beam_on_sp_comm_read"`; 
do 
    sim_id=`echo $i | grep -o 'sim[[:digit:]]\+'`;
    mic_id=`basename $i .wav`;
    room_id=`echo $i | grep -o 'Kitchen\|Livingroom'`;
    new_id=dev1_${sim_id}_${room_id}_beam
    new_audio_file_name=audio/${new_id}.wav
    new_lab_file_name=lab/${new_id}.lab
    
 
    ln -s $i $new_audio_file_name;
 #  # ln -s $lab_dir/${mic_id}/${mic_id}`echo $sim_id | sed s/sim//`.lab $new_lab_file_name; 
    ln -s $lab_dir/dev1_${sim_id}_${room_id}.lab $new_lab_file_name;
    echo $new_audio_file_name $new_lab_file_name >> $training_script;
done;

lab_dir=/rmt/work/speech_activity_detection_git/lab/transcr_beam/test1
for i in `ls /rmt5/databases/DIRHA/DIRHA_simcorpora2_v3/GR/test1/simulations/sim*/Signals/Mixed_Sources/*/*/*.wav | grep "beam_on_sp_comm_read"`; 
do 
    sim_id=`echo $i | grep -o 'sim[[:digit:]]\+'`;
    mic_id=`basename $i .wav`;
    room_id=`echo $i | grep -o 'Kitchen\|Livingroom'`;
    new_id=test1_${sim_id}_${room_id}_beam
    new_audio_file_name=audio/${new_id}.wav
    new_lab_file_name=lab/${new_id}.lab

    ln -s $i $new_audio_file_name;
   # ln -s $lab_dir/${mic_id}/${mic_id}`echo $sim_id | sed s/sim//`.lab $new_lab_file_name;
    ln -s $lab_dir/test1_${sim_id}_${room_id}.lab $new_lab_file_name; 
    echo $new_audio_file_name $new_lab_file_name >> $testing_script;

  #  echo "something found..";
done;



