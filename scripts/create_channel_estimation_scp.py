import os

root_dir = '/home/work/speech_activity_detection/'
all_files_scp= os.path.join(root_dir, 'lists', 'test_dryrun.scp')
channel_estimation_scp = os.path.join(root_dir, 'lists', 'channel_estimation_dry_run.scp')

af_scp = open(all_files_scp, 'r')
ce_scp = open(channel_estimation_scp, 'w')
for ln in af_scp:
    f_name = ln.split()[1]
    b_name = os.path.splitext(os.path.split(f_name)[1])[0]
    b_info = b_name.split('_')
    channel_name = b_info[-1]
    print channel_name
    ce_scp.write('{} {}\n'.format(f_name, channel_name))

af_scp.close()
ce_scp.close()
