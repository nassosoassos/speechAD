root_dir=/home/work/speech_activity_detection
export PYTHONPATH=${root_dir}/scripts/PLP_VAD/python_src
for channel in {B,E,F}
do
  working_dir=${root_dir}/experiments/test_${channel}_dry_run
  python PLP_VAD/python_src/plp_vad_gmm/darpa_experiments.py -c ${root_dir}/scripts/experiments.cfg --train_script ${root_dir}/lists/${channel}_train.scp --test_script ${root_dir}/lists/${channel}_test_dryrun.scp --working_dir ${working_dir} && 
  rm -rf ${working_dir}/PLP_*
done
