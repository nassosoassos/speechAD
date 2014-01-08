root_dir=/home/work/speech_activity_detection
export PYTHONPATH=${root_dir}/scripts/PLP_VAD/python_src
channel=B
working_dir=${root_dir}/experiments/test_${channel}
for acc in {5,7,9,11,13,15,21,25,31}
do
  for sh in {15,10,5}
    do
      let hlda_nuisance_dims=($acc-3)*13 
      for ngmm in {4,8,16,32,64,128,256}
        do
          echo $acc $ngmm $sh
          python PLP_VAD/python_src/plp_vad_gmm/darpa_experiments.py -c ${root_dir}/scripts/experiments.cfg --train_script ${root_dir}/lists/${channel}_train_sample.scp --test_script ${root_dir}/lists/${channel}_test_sample.scp --working_dir ${working_dir} --acc_frame_shift $sh --n_gmm_components $ngmm --acc_frames $acc --hlda_nuisance_dims $hlda_nuisance_dims
          tail -n 100 $working_dir/experiment.log | grep -A 3 java 
        done
    done
done
