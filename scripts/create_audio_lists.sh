root_dir=/home/work/speech_activity_detection
data_dir=$root_dir
export PYTHONPATH=${root_dir}/scripts/PLP_VAD/python_src
for i in {B,E,F}
do
ls $data_dir/sad_ldc2011e111/sad-part3/data/train/*/audio/${i}_16000/*.wav > ${root_dir}/lists/${i}_train_audio.list
ls $data_dir/sad_ldc*/data/train/*/audio/${i}_16000/*.wav >> ${root_dir}/lists/${i}_train_audio.list
cat ${root_dir}/lists/${i}_AllSadDryRun?FileWavAnnot > ${root_dir}/lists/${i}_AllSadDryRunFileWavAnnot
cat ${root_dir}/lists/${i}_AllSadDryRunFileWavAnnot | cut -f 3 | sed 's/\/media\/DATTRANSFER\/RATS\/annotations/\/home\/work\/speech_activity_detection/' > ${root_dir}/lists/${i}_test_trans.list
#python ${root_dir}/scripts/PLP_VAD/python_src/my_utils/find_corresponding_files.py --in_list ${root_dir}/lists/${i}_train_audio.list --search_dir ${data_dir}/LDC2011E100 --search_dir ${data_dir}/LDC2011E112 --search_dir ${data_dir}/LDC2011E87 --suffix txt --scp_file tmp.scp
#cat tmp.scp | uniq > ${root_dir}/lists/${i}_train.scp
python ${root_dir}/scripts/PLP_VAD/python_src/my_utils/find_corresponding_files.py --in_list ${root_dir}/lists/${i}_test_trans.list --search_dir ${data_dir}/sad_ldc2011e86_v2 --search_dir ${data_dir}/sad_ldc2011e99 --search_dir ${data_dir}/sad_ldc2011e111 --suffix wav --reverse --scp_file tmp.scp 
cat tmp.scp | uniq > ${root_dir}/lists/${i}_test_dryrun.scp
done
