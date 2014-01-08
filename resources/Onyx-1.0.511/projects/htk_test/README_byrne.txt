===========================================================================
=
= File:         README_byrne.txt (directory: projects/htk_test)
= Date:         2008-11-25 Tue 18:42:02
= Author:       Ken Basye
= Description:  Recipe for HTK model building for ARPA RM task
=
= This file is part of Onyx   http://onyxtools.sourceforge.net
=
= Copyright 2008, 2009 The Johns Hopkins University
=
= Licensed under the Apache License, Version 2.0 (the "License").
= You may not use this file except in compliance with the License.
= You may obtain a copy of the License at
=   http://www.apache.org/licenses/LICENSE-2.0
= 
= Unless required by applicable law or agreed to in writing, software
= distributed under the License is distributed on an "AS IS" BASIS,
= WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
= implied.  See the License for the specific language governing
= permissions and limitations under the License.
=
===========================================================================

The following document is a guide to how to build HTK based systems
for the ARPA RM task.

The systems described have not necessarily been optimised and so the
results should not be taken as 'the best that can be done' but more as
a baseline. Note that most of the HTK research work at CUED involves
continuous density models and little effort has been put in to
optimising the performance of the discrete and tied mixture systems
and so these are definitely naive baseline results.

Note.  This is not a guide to setting up HTK and for the recipe to work
       HTK must be correctly installed and all the executables in the users
       path.

CLSP Users:
-----------
This is a guide to using the Resource Management Toolkit at CLSP.
Much of the data files and library files have already been created, so
please don't duplicate these files unnecessarily.  The distributed
version of this README file is in  /home/byrne/RMHTK_V3.0/README .

The HTK Book is online at /opt/HTK-3.2/HTKBook/ (on the Linux cluster).

---------------------------------------------------------------------


                          RM RECIPE
                          =========

The environment variables RMDATA, and RMLIB should be set as
follows:
 > setenv RMLIB /export/byrne/RMHTK_V3.0/lib
 > setenv RMDATA /export/data02/RMTK/mfcc
Make a directory in which you'll do your work.  For example,
you could make a directory in your home :
 > mkdir ~/rmwork
Set the environment variable RMWORK to point to your work directory, e.g.
 > setenv RMWORK ~/rmwork
Copy the contents of /home/byrne/RMHTK_V3.0/work/ to your work directory :
 > cp -r /home/byrne/RMHTK_V3.0/work/ $RMWORK

The various HTK scripts as well as all the HTK tools should be in your
current path. The scripts are in the directory
 /home/byrne/RMHTK_V3.0/scripts

If you're using csh or tcsh,  the following should set your path:
 > set path = (/home/byrne/RMHTK_V3.0/scripts $path)
Use one of the linux cluster machines, and set the following
 > set path = (/opt/HTK-3.2/bin.linux_glibc/ $path)

The recipe provides instructions for creating several types of
systems.

1)  Single mixture monophones **
2)  Multiple mixture monophones  [from 1] **
3)  Tied-mixture monophones [from 1 or 2]
4)  Discrete density monophones [with data aligned by 1 or 2]
5)  Single mixture word internal triphones [from 1]  ** (SKIP 5.6)
6)  Tied mixture word internal triphones [from 2]
7)  Single mixture cross word triphones [from 1]
8)  Tied state triphones (data driven clustering) [from 5]
9)  Tied state triphones (decision tree clustering) [from 5 or 7]  **
10) Bigram generation and testing.

As this shows there are many routes through this demo.
These instruction will explain (and give results for) the sequences
1 -> 2 -> 3,
1 -> 4,
1 -> 2 -> 6,
1 -> 5 -> 8,
1 -> 7 -> 9

The following instructions should be followed exactly as given, with
close attention paid to the naming conventions for files and directory
hierarchies.


1. Creating a Base-Line Monophone Set
=====================================

Note: This recipe skips Steps 1.1 and 1.2 .
The material you need is in /export/byrne/RMHTK_V3.0/lib .

1.3  Initial phone models

The supplied initial models should be in the directory $RMWORK/R1/hmm0.
Note that the MODELS file also contains a varFloor vector which
was calculated by using HCompV

 > HCompV -C $RMLIB/configs/config.basic -S $RMLIB/flists/ind_trn109.scp \
    -f 0.01 -m varProto

The supplied HTK Environment file HTE should also be in $RMWORK/R1.
This contains a number of shell variables that are used by the
various training and testing scripts.  You shouldn't need to edit
this file yet if you've set your environment variables correctly.

1.4  Model training

Re-estimate the models in hmm0 for four iterations of HERest.

 > cd $RMWORK/R1
 > hbuild 1 4

This should create the directories $RMWORK/R1/hmm[1-4]. Each of these
directories should contain a LOG file and also $RMWORK/R1 will contain
a file called blog which records the build process. After running
hbuild the various LOG files should be checked for error messages.


1.5  Building the recognition networks
<You can skip 1.5  . Use the files in $RMLIB/nets >

Copy the word-pair grammar from the CD-ROM to $RMLIB.

 > cd $RMLIB; mkdir nets
 > cp $RMCD2/rm1/doc/wp_gram.txt $RMLIB/nets

Use the supplied gawk scripts to generate word level networks for
both the word-pair and no-grammar test conditions.  Note that the
language model likelihoods are based solely on the number of successors
for each word.

 > cd $RMLIB/nets
 > cat wp_gram.txt wp_gram.txt | fgrep -v '*' | \
    gawk -f $RMLIB/awks/wp.net.awk nn=993 nl=57704 > net.wp
 > gawk -f $RMLIB/awks/ng.net.awk nn=994 nl=1985 $RMLIB/wordlist > net.ng
 > gzip net.wp net.ng


1.6  Testing the Monophone Models

Check that all the recognition variables in HTE are set up correctly.
The recognition tests use the script htestrm. The recognition
parameters are set in the HTE file as for hbuild.  htestrm allows the
specification of the test set, the recognition mode (ng or wp) and
automatically creates test directories for a number of runs (typically
with different recognition parameters and perhaps run simultaneously
on different machines). Test directories are created as
sub-directories of the model directories.  The first run on a
particular hmm set with ng on the feb89 test set would be called
test_ng_feb89.1, the second run test_ng_feb89.2, and tests with wp and
other test-sets named according to the above convention.

A number of parameters can be tuned including the grammar scale factor
HVGSCALE variable (HVite -s) and inter-word probability HVIMPROB
(HVite -p).  These flags both control the ratio of insertion and
deletion errors.  HVite pruning is controlled by variables HVPRUNE
(HVite -t) and HVMAXACTIVE (HVite -u). All of these values are
supplied in an indexed list that correspond to the different test types
listed in the HTE variable TYPELIST (here either ng or wp). The
HVPRUNE values should be set at a higher value for wp testing than ng
testing.

After htestrm runs HVite, HResults is executed. The equivalent sets
(homophone lists) for the different test conditions are listed in
files specified by the HREQSETS variable. These sets are defined by
DARPA/NIST and suitable files (eq.wp and eq.ng) are supplied with the
distribution. If the variable HRNIST is set HResults operates in a
manner that is compatible with the NIST scoring software and results
identical to those from the NIST software should be obtained.

To test the models in $RMWORK/R1/hmm4 with the feb89 test set and with
a word-pair grammar and the current recognition settings in HTE,
execute the following

 > cd $RMWORK/R1
 > htestrm HTE wp feb89 hmm4

Run other tests under a number of ng/wp conditions on different test
sets as desired.


2. Multiple Mixture Monophones
==============================

2.1  Mixture splitting

The models created in Step 1 will be converted to multiple mixture
models by iterative "mixture splitting" and re-training.
Mixture-splitting is performed by HHEd by taking an output distribution
with M output components, choosing the components with the largest
mixture weights and making two mixtures where there was one before by
perturbing the mean values of the split mixture components. This can
be done on a state-by-state basis but here all states will be mixture
split.

A script interface to HHEd called hedit takes a script of HHEd
commands and applies them. First two-mixture models will be created,
with the initial models in $RMWORK/R2/hmm10 formed by
mixture-splitting the models in $RMWORK/R1/hmm4.

First create the directory to hold the multiple mixture monophones and
link all the files and directories to the new directory.

 > mkdir $RMWORK/R2
 > cd $RMWORK/R2
 > ln -s $RMWORK/R1/* .

Now store the HHEd commands required in a file called edfile4.10 in
$RMWORK/R2. This file need contain only the line

 MU 2 {*.state[2-4].mix}

This command this will split into 2 mixtures all the states of all
models in the HMMLIST defined in the HTE file. Now invoke hedit by

 > cd $RMWORK/R2
 > hedit 4 10

This creates 2 mixture initial hmms in $RMWORK/R2/hmm10.

2.2  Retraining and testing the two-mixture models

Run four iterations of  HERest to re-train the models

 > cd $RMWORK/R2
 > hbuild 11 14

and then test them with HVite (again recognition settings may need
adjusting).

 > cd $RMWORK/R2
 > htestrm HTE wp feb89 hmm14

and run other tests as required.

2.3  Iterative mixture-splitting

The mixture splitting and re-training can be done repeatedly and it is
suggested that after the 2 mixture models are trained in order 3
mixture, 5 mixture 7 mixture, 10 mixture and 15 mixture models are
trained.  Each cycle requires forming a file containing the
appropriate HHEd MU command, running hedit, running hbuild and then
testing the models with htestrm.

Note that it is quite possible to go straight from 1 mixture models to
5 mixture models. However usually better performance results from a
given number of mixture components if the model complexity is
increased in stages.

3. Tied Mixture Monophones
==========================

3.1  Tied Mixture Initialisation

The models created in step 2 will be converted to a set of tied-mixture
models using HHEd.  This conversion is performed in three phases.
First the set of Gaussians that represent the models are pooled by
choosing those with the highest weights.  Then the models are rebuilt
to share these Gaussians and the mixture weights recalculated.
Finally the representation of the models changed to TIEDHS.

First create the directory to hold the tied mixture monophones and
copy over the HTE file.

 > mkdir $RMWORK/R3
 > cd $RMWORK
 > cp R2/HTE R3

Create an HHEd edit file in $RMWORK/R3 called tied.hed containing the
following commands.

 JO 128 2.0
 TI MIX_ {*.state[2-4].stream[1].mix}
 HK TIEDHS

Create a directory for the initial tied mixture models and run HHEd.

 > cd $RMWORK/R3
 > mkdir hmm0
 > HHEd -T 1 -H $RMWORK/R2/hmm4/MODELS -w hmm0/MODELS tied.hed \
    $RMLIB/mlists/mono.list

3.2  Tied Mixture training and testing

In the HTE file the line

 set TMTHRESH=20

now controls tied mixture as well as forward pass mixture pruning.
Finally build the models

 > cd $RMWORK/R3
 > hbuild 1 4

and then test them with HVite (recognition settings will need adjusting.
Try setting the grammar scale to 2.0 rather than 7.0 and speed things up
a little by changing the pruning beam width from 200.0 to 100.0).

 > cd $RMWORK/R3
 > htestrm HTE wp feb89 hmm4


4. Discrete Density Monophones
==============================

4.1  Vector quantiser creation

Since all the observation have to be held in memory for this operation
(which can be computationally expensive) we will perform this on a
subset of the data. Please note that this process can take a considerable
amount of time.

 > mkdir $RMWORK/R4
 > cd $RMWORK/R4
 > gawk '{ if ((++i%10)==1) print }' $RMLIB/flists/ind_trn109.scp > ind_trn109.sub.scp
 > HQuant -A -T 1 -d -n 1 128 -n 2 128 -n 3 128 -n 4 32 -s 4 -S \
     ind_trn109.sub.scp -C $RMLIB/configs/config.basic vq.table

4.2  Align data and initialise discrete models

To create some discrete models we are going to use HInit and HRest both
of which need data aligned at the phone level.  We will use the monophones
from either R1 or R2 to generate phone labels for the subset of the data
used for the codebook.

 > HVite -A -T 1 -C $RMLIB/configs/config.basic -H ../R1/hmm4/MODELS \
    -a -m -o SW -i ind_trn109.sub.mlf -X lab -b '!SENT_START' \
    -l '*' -y lab -I $RMLIB/wlabs/ind_trn109.mlf -t 500.0 -s 0.0 -p 0.0 \
    -S ind_trn109.sub.scp $RMLIB/dicts/mono.dct $RMLIB/mlists/mono.list

4.3  Create a prototype hmm set

This can be done using the MakeProtoHMMSet discrete.pcf or by creating
a file like the one below for each of the models apart from sp which should
only have three states so that is reproduced in full.

  ~o <VecSize> 39 <MFCC_E_D_A_V> <StreamInfo> 4 12 12 12 3
  ~h "aa"
<BeginHMM>
  <NumStates> 5
  <State> 2 <NumMixes> 128 128 128 32
  <Stream> 1
      <DProb> 11508*128
  <Stream> 2
      <DProb> 11508*128
  <Stream> 3
      <DProb> 11508*128
  <Stream> 4
      <DProb> 8220*32

  <State> 3 <NumMixes> 128 128 128 32
  <Stream> 1
      <DProb> 11508*128
  <Stream> 2
      <DProb> 11508*128
  <Stream> 3
      <DProb> 11508*128
  <Stream> 4
      <DProb> 8220*32

  <State> 4 <NumMixes> 128 128 128 32
  <Stream> 1
      <DProb> 11508*128
  <Stream> 2
      <DProb> 11508*128
  <Stream> 3
      <DProb> 11508*128
  <Stream> 4
      <DProb> 8220*32

  <TransP> 5
   0.000e+0   1.000e+0   0.000e+0   0.000e+0   0.000e+0
   0.000e+0   6.000e-1   4.000e-1   0.000e+0   0.000e+0
   0.000e+0   0.000e+0   6.000e-1   4.000e-1   0.000e+0
   0.000e+0   0.000e+0   0.000e+0   6.000e-1   4.000e-1
   0.000e+0   0.000e+0   0.000e+0   0.000e+0   0.000e+0
<EndHMM>

  ~o <VecSize> 39 <MFCC_E_D_A_V> <StreamInfo> 4 12 12 12 3
  ~h "sp"
<BeginHMM>
  <NumStates> 3
  <State> 2 <NumMixes> 128 128 128 32
  <Stream> 1
      <DProb> 11508*128
  <Stream> 2
      <DProb> 11508*128
  <Stream> 3
      <DProb> 11508*128
  <Stream> 4
      <DProb> 8220*32

  <TransP> 3
   0.000e+0   6.000e-1   4.000e-1
   0.000e+0   6.000e-1   4.000e-1
   0.000e+0   0.000e+0   0.000e+0
<EndHMM>

 > mkdir hmm0
 > MakeProtoHMMSet $RMLIB/discrete.pcf

Then initialise the models. Note that the following script commands
assume that you running csh unix shell. These commands vary for different
shells.

 > mkdir hmm1
 > foreach i (`cat $RMLIB/mlists/mono.list`)
   HInit -T 1 -C $RMLIB/configs/config.discrete -M hmm1 -l $i \
     -I ind_trn109.sub.mlf -S ind_trn109.sub.scp hmm0/$i
   end

And reestimate them.

 > mkdir hmm2
 > foreach i (`cat mono.list`)
   HRest -T 1 -C $RMLIB/configs/config.discrete -M hmm2 -l $i \
     -I ind_trn109.sub.mlf -S ind_trn109.sub.scp hmm1/$i
   end

Create an HHEd script edfile2.3 with the single command

  AT 1 3 0.8 { sp.transP }

This will add back the tee-transition in sp deleted by HInit and
HRest and will also produce a single MMF

 > mkdir hmm3
 > HHEd -A -T 1 -d hmm2 -w hmm3/MODELS edfile2.3 $RMLIB/mlists/mono.list

Copy the standard HTE file from the R1 directory and change the
configuration file names to use the discrete configuration file.

setenv HECONFIG $rmlib/configs/config.discrete
setenv HVCONFIG $rmlib/configs/config.discrete

Then train the models

 > hbuild 4 7

and then test them with HVite (recognition settings will need adjusting.
Try setting the grammar scale to 2.0 rather than 7.0 and speed things up
a little by changing the pruning beam width from 200.0 to 100.0).

 > cd $RMWORK/R4
 > htestrm HTE wp feb89 hmm7


5. Single-Mixture Word-Internal Triphones
=========================================

5.1  Triphone dictionary and model-list creation

First a new dictionary is needed. This can be created using the
tool HDMan with the supplied script tri.ded. This creates
word-internal triphones for each word in the dictionary and creates a
context-dependent model list as a by-product.

The dictionary you need can be found in
 /export/byrne/RMHTK_V3.0/lib/dicts/tri.dct

5.2  Triphone training label files

The triphone phone-level training label file you need
can be found in /export/byrne/RMHTK_V3.0/lib/labs/tri.mlf .

5.3  Initial models

Next an initial set of triphone models are created by cloning the
monophone models using HHEd. First create a directory with a copy of
R1/HTE.

 > cd $RMWORK; mkdir R5
 > cp R1/HTE R5

Edit R5/HTE to change the title, HMMLIST, TRAINMLF and HVVOC
parameters replacing mono with tri.

For convenience, create local copies (links) of both the monophone
list and the triphone list.

 > cd $RMWORK/R5
 > ln -s $RMLIB/mlists/mono.list mono.list
 > ln -s $RMLIB/mlists/tri.list tri.list

Create an HHEd edit file in $RMWORK/R5 called clone.hed containing the
following commands.

 MM "trP_" { *.transP }
 CL "tri.list"

Create a directory for the initial cloned triphones and run HHEd.

 > mkdir hmm0
 > HHEd -B -T 1 -H $RMWORK/R1/hmm4/MODELS -w hmm0/MODELS clone.hed mono.list


5.4  Triphone training

The HERest pruning threshold should be increased for triphone models.
In HTE, change the value of HEPRUNE to 1000.0. Then build the new set
of models using hbuild.
Note that for future use (state-clustering) a statistics file for the
final run should be created. This is achieved by the following line
in the HTE file.

 set HESTATS=stats

 > cd $RMWORK/R5
 > hbuild 1 2

5.5  Triphone Testing

Test the models using  htestrm as usual.

 > htestrm HTE wp feb89 hmm2

It may be found that it is necessary to adjust some of the word-insertion or
pruning penalties for triphone models.

5.6  Adaptation (to speaker dms0)

First, we should see what results are like for speaker dms0 before adaptation takes place

 > htestrm HTE wp dms0_tst hmm2

The results should be (depending upon pruning thresholds):

HResults -A -z ::: -I /import/home/dp/RMDEMO/lib/wlabs/dms0_tst.mlf -n -e ::: !SENT_START -e ::: !SENT_END /import/home/dp/RMDEMO/lib/wordlist hmm2/dms0_tst/dms0_tst.mlf
====================== HTK Results Analysis =======================
  Date: Mon Jan 11 09:36:08 1999
  Ref : /import/home/dp/RMDEMO/lib/wlabs/dms0_tst.mlf
  Rec : hmm2/dms0_tst/dms0_tst.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=45.92 [H=45, S=53, N=98]
WORD: %Corr=88.36, Acc=85.81 [H=691, D=26, S=65, I=20, N=782]
===================================================================

Now we should do the adaptation.  First, build a regression class tree using HHEd,
then generate the transforms (done in 2 stages,  global first, then using
regression class tree)

A few speaker dependent data files may contain errors, but a partial hypothesis is
fine for testing purposes. To prevent errors later the FORCEOUT parameter is
set to true in the adaptation config file.

 > HVite -A -H hmm2/MODELS -T 1 -I $RMLIB/wlabs/traina.mlf -C $RMLIB/configs/config.adapt
-l '*' -o SWT -a -m -t 400.0 -y lab -i adaptPhones.mlf -S $RMLIB/flists/traina.scp
$RMLIB/dicts/mono.dct $RMLIB/mlists/tri.list

 > echo IS sil sil > sil.hled
 > HLEd -l '*' -i adaptPhones_sil.mlf -d $RMLIB/dicts/tri.dct sil.hled adaptPhones.mlf

 > mkdir hmm3
 > echo LS "hmm2/stats" > regtree.hed
 > echo RC 32 "rtree" >> regtree.hed

 > HHEd -H hmm2/MODELS -M hmm3 regtree.hed tri.list

 > HEAdapt -C $RMLIB/configs/config.adapt -g -S $RMLIB/flists/traina.scp -I adaptPhones_sil.mlf -H hmm3/MODELS -K hmm3/global.tmf tri.list
 > HEAdapt -C $RMLIB/configs/config.adapt -S $RMLIB/flists/traina.scp -I adaptPhones_sil.mlf -H hmm3/MODELS -J hmm3/global.tmf -K hmm3/rc.tmf tri.list

Before running the test, add the following into the HTE file to pick up the new transformation:

 set HVTRANSFORM=hmm3/rc.tmf

Now do the evaluation:

 > htestrm HTE wp dms0_tst hmm3

The results should now look like:

HResults -A -z ::: -I /import/home/dp/RMDEMO/lib/wlabs/dms0_tst.mlf -n -e ::: !SENT_START -e ::: !SENT_END /import/home/dp/RMDEMO/lib/wordlist hmm3/dms0_tst/dms0_tst.mlf
====================== HTK Results Analysis =======================
  Date: Mon Jan 11 16:22:12 1999
  Ref : /import/home/dp/RMDEMO/lib/wlabs/dms0_tst.mlf
  Rec : hmm3/dms0_tst/dms0_tst.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=66.33 [H=65, S=33, N=98]
WORD: %Corr=94.64, Acc=92.85 [H=741, D=10, S=32, I=14, N=783]
===================================================================


6. Tied-Mixture Word-Internal Triphones
=======================================

6.1  Tied Mixture Initialisation

The models created in step 2 will be converted to a set of triphones
in a similar way to that used in step 5.
At the same time the single stream continuous models will be converted
to tied-mixture and expanded to use four streams.

First create an initial set of models.

 > mkdir $RMWORK/R6
 > cd $RMWORK
 > cp R2/HTE R6
 > cd $RMWORK/R6
 > ln -s $RMLIB/mlists/mono.list mono.list
 > ln -s $RMLIB/mlists/tri.list tri.list

Create an HHEd edit file in $RMWORK/R6 called clone.hed containing the
following commands.

 MM "trP_" { *.transP }
 SS 4
 JO 128 2.0
 TI MIX_1_ {*.state[2-4].stream[1].mix}
 JO 128 2.0
 TI MIX_2_ {*.state[2-4].stream[2].mix}
 JO 128 2.0
 TI MIX_3_ {*.state[2-4].stream[3].mix}
 JO 32 2.0
 TI MIX_4_ {*.state[2-4].stream[4].mix}
 HK TIEDHS
 CL "tri.list"


Create a directory for the initial cloned triphones and run HHEd.

 > mkdir hmm0
 > HHEd -B -T 1 -H $RMWORK/R2/hmm4/MODELS -w hmm0/MODELS clone.hed mono.list

Then train the models

 > hbuild 1 2

and test them using  htestrm as usual.

 > htestrm HTE wp feb89 hmm2

As a final stage to training it is possible to use deleted interpolation
smoothing to provide more robust estimates for the mixture weights
using HSmooth as the final stage of a parallel mode run of HERest

 > foreach i ( 1 2 3 4 )
   gawk '{ if ((++i%4)==(n%4)) print }' n=$i $RMLIB/flists/ind_trn109.scp \
    > ind_trn109.$i.scp
   end
 > mkdir hmm3
 > foreach i ( 1 2 3 4 )
   HERest -A -H hmm2/MODELS -M hmm3 -T 1 -I $RMLIB/labs/tri.mlf \
    -w 2.0 -t 600.0 -c 20.0 -C $RMLIB/configs/config.basic \
    -p $i -S ind_trn109.$i.scp $RMLIB/mlists/tri.list > hmm3/LOG$i
   end
 > HSmooth -A -H hmm2/MODELS -M hmm3 -T 1 -m 1 -w 2.0 -s hmm3/stats \
    -C $RMLIB/configs/config.basic $RMLIB/mlists/tri.list hmm3/HER*

Finally test these models as usual

 > htestrm HTE wp feb89 hmm3


7. Single-Mixture Cross-Word Triphones
======================================

7.1  Triphone model-list and training label file creation

First a set of cross word labels and a model list is needed for
training.
These are generated using HLEd and the monophone label file.

 > HLEd -l '*' -n $RMLIB/mlists/xwrd.list -i $RMLIB/labs/xwrd.mlf \
    $RMLIB/labs/xwrd.hled $RMLIB/labs/mono.mlf

Note that this model list will not contain many of the triphones
needed for testing (especially in the ng case).  These models
will have to be synthesised later.


7.2  Initial models

Next an initial set of triphone models are created by cloning the
monophone models using HHEd. First create a directory with a copy of
R1/HTE.

 > cd $RMWORK; mkdir R7
 > cp R1/HTE R7

Edit R7/HTE to change the title, HMMLIST and TRAINMLF
parameters replacing mono with xwrd.

For convenience, create local copies (links) of both the monophone
list and the triphone list.

 > cd $RMWORK/R7
 > ln -s $RMLIB/mlists/mono.list mono.list
 > ln -s $RMLIB/mlists/xwrd.list xwrd.list

Create an HHEd edit file in $RMWORK/R7 called clone.hed containing the
following commands.

 MM "trP_" { *.transP }
 CL "xwrd.list"

Create a directory for the initial cloned triphones and run HHEd.

 > mkdir hmm0
 > HHEd -B -T 1 -H $RMWORK/R1/hmm4/MODELS -w hmm0/MODELS clone.hed mono.list

Seeing this model set can be very large it is best to save it in binary
format by using the -B flag in the above command and adding

 > HMODEL: SAVEBINARY = T

to the training configuration file $RMLIB/configs/config.basic

7.3  Triphone training

The HERest pruning threshold should be increased for triphone models.
In HTE, change the value of HEPRUNE to 1000.0. Then build the new set
of models using hbuild.
Note that for future use (state-clustering) a statistics file for the
final run should be created and the models should be updated even
if they only occur a few times. This is achieved by the following
lines in HTE.

 set HESTATS=stats
 set HEMINEG=0

 > cd $RMWORK/R7
 > hbuild 1 2

7.4  Cross-Word Triphone Testing

As mentioned above due to lack of coverage of the test condition
because of missing models it is impractical to test the unclustered
cross word triphone system.  If the model list used for the CL
command above contained all models needed for recognition testing
would be possible however many of the models would still be using
monophone parameters.


8. State-Clustered Triphones
============================

8.1  State clustering

In this section the triphones created in Step 5 will be state
clustered, so that all states within the same cluster share a common
output distribution. Corresponding states of different triphones of the
same phone are candidates to be put in the same cluster.  The
clustering process groups states with similar distributions and also
ensures (via the HHEd RO command) that any states with too few
occupations to allow reliable estimates of a multiple mixture
distribution are discarded.

8.2  Initial models

The clustering process is performed by HHEd. To simplify preparing
the HHEd script, mkclscript does most of the work.  Given the
monophone model list it generates for each model commands to tie all
the transition matrices of each triphone together and also to
cluster the states. It is assumed that each model in the list has
three states.

However, the clustering commands should not be applied to the sil and
sp models. First make a lost copy of mono.list and delete the entries
for sp and sil, copy the HTE file from R5 and change the HMMLIST to use
the clustered triphone list trig.list rather than the compete list
tri.list

 > mkdir $RMWORK/R8
 > cd $RMWORK/R8
 > cp $RMWORK/R5/HTE .
 > egrep -v 'sil|sp' $RMLIB/mlists/mono.list > mono.list

Use the mkclscript to create the HHEd script

 > echo 'RO 100.0 stats' > cluster.hed
 > mkclscript TC 0.5 mono.list >> cluster.hed
 > echo 'CO "trig.list"' >> cluster.hed

These lines tell HHEd to allow a minimum number of state occupations of
100 and to compact the model set so that identical logical
models share the same physical model list.  Also make the final stats
stats generated in R5 present in the current directory by executing

 > ln -s ../R5/hmm2/stats stats

Make a directory for the new state-clustered models and run HHEd

 > cd $RMWORK/R8
 > mkdir hmm0
 > HHEd -T 1 -H $RMWORK/R5/hmm2/MODELS -w hmm0/MODELS cluster.hed \
    $RMLIB/mlists/tri.list > clog

and copy the HMM list created to $RMLIB/mlists.

 > cp trig.list $RMLIB/mlists


8.3  Building state-clustered models

Copy $RMWORK/R5/HTE to the R4 directory, edit the title, and change
the HMMLIST variable to $RMLIB/mlists/trig.list.
Now build a set of single mixture-state clustered triphones

 > hbuild 1 4

and test them as before.

 > htestrm HTE wp feb89 hmm4


8.4  Multiple mixture state-clustered triphones

Multiple mixture models for the state-clustered triphones are built
exactly as for the monophone multiple mixture models. However in this
case since there are more output distributions models with a smaller
number of mixtures/distribution are built.  It is suggested that 2
mixture, 3 mixture, 4 mixture and then 5 mixture models be built, and
at each stage 4 iterations of HERest be performed.

To obtain the initial 2 mixture state-clustered triphones create the
file $RMWORK/R4/edfile4.10 containing the line

 MU 2 {*.state[2-4].mix}

run hedit to build a new set of models in hmm10 and then hbuild

 > hedit 4 10
 > hbuild 11 14

etc., until the 5 mixture models are built.

After each set of models have been build they can be tested using
htestrm as usual.

9. Tree-Clustered Tied-State Triphones
======================================

9.1  State clustering

In this section the triphones created in Step 5 or 7 will be state
clustered.  However rather than using the data-driven method of
state clustering used in Step 8 a decision tree based one is used.
This allows the synthesis of unseen triphones and thus makes it
possible to produce cross-word context dependent systems.
The clustering is used in a very similar way to that in Step 8 with
sharing only possible within the same state of the same base phone.
However the clustering proceeds in a top down manner by initially
grouping all contexts and then splitting on the basis of questions
about context.  The questions used are chosen to maximise the
likelihood of the training data whilst ensuring that each tied-state
has a minimum occupancy (again using the HHEd RO command).


9.2  Initial models

The clustering process is performed by HHEd. To simplify preparing
the HHEd script, mkclscript does most of the work and an example
set of questions are supplied with the demo.  Given the
monophone model list it generates for each model commands to tie all
the transition matrices of each triphone together and also to
cluster the states. It is assumed that each model in the list has
three states.

However, the clustering commands should not be applied to the sil and
sp models. First make a lost copy of mono.list and delete the entries
for sp and sil.

 > mkdir $RMWORK/R9
 > cd $RMWORK/R9
 > egrep -v 'sil|sp' $RMLIB/mlists/mono.list > mono.list

We also need to generate a list of the complete set of models needed
during recognition.
If we are still using word internal models (from R5) we just use the
same triphone list

 > set src = R5
 > set list = tri.list
 > cp $RMLIB/mlists/tri.list unseen.list

% Note: Skip this step unless you're training cross word triphones.
%
%However for a cross-word system there are many contexts that we
%have not seen that can occur in our recognition networks.
%Rather than actually find out which models are needed it is
%easier to generate all possible monophones, biphones and triphones
%and this would also allow us to work with an arbitrary vocabulary.
%
% > set src = R7
% > set list = xwrd.list
% > awk -f $RMLIB/awks/full.list.awk mono.list > unseen.list

Use the mkclscript to create the HHEd script

 > echo 'RO 100.0 stats' > cluster.hed
 > cat $RMLIB/quests.hed >> cluster.hed
 > mkclscript TB 600.0 mono.list >> cluster.hed
 > echo 'ST "trees"' >> cluster.hed
 > echo 'AU "unseen.list"' >> cluster.hed
 > echo 'CO "treeg.list"' >> cluster.hed

These lines tell HHEd to allow a minimum number of state occupations of
100 and to compact the model set so that identical logical
models share the same physical model list.  Also make the file stats
generated in R5 or R7 present in the current directory by executing

 > ln -s ../$src/hmm2/stats stats

Make a directory for the new state-clustered models and run HHEd

 > cd $RMWORK/R9
 > mkdir hmm0
 > HHEd -T 1 -B -H $RMWORK/$src/hmm2/MODELS -w hmm0/MODELS cluster.hed \
    $RMWORK/$src/$list > clog

9.3  Building state-clustered models

Copy $RMWORK/$src/HTE to the R9 directory, edit the title, and change
the HMMLIST variable to $rmwork/R9/treeg.list
It is also necessary to increase the pruning beam width somewhat as
well as increase the word insertion penalty

% Note: Skip this step unless you're training cross-word triphones.
% setenv HVCONFIG $rmlib/configs/config.xwrd

set HVPRUNE=(200.0 300.0)

set HVIMPROB=(-40.0 -40.0)

Now build a set of single mixture-state clustered triphones

 > hbuild 1 4

and test them as before.

 > htestrm HTE wp feb89 hmm4


9.4  Multiple mixture state-clustered triphones

Multiple mixture models for the state-clustered cross-word triphones
are built exactly as for the word-internal triphone models.
It is suggested that 2 mixture, 3 mixture, 4 mixture, 5 mixture and
then 6 mixture models be built, and at each stage 4 iterations of
HERest be performed.

To obtain the initial 2 mixture state-clustered triphones create the
file $RMWORK/R9/edfile4.10 containing the line

 MU 2 {*.state[2-4].mix}

run hedit to build a new set of models in hmm10 and then hbuild

 > hedit 4 10
 > hbuild 11 14

etc., until the 6 mixture models are built.

After each set of models have been build they can be tested using
htestrm as usual.

NOTE:  Due to the 'two' pronunciations for each word, one ending
       in sil and the other in sp, it is possible to get errors
       due to no token reaching the end of the network when sp
       is a significantly better model than sil.  These can be
       avoided by raising the beam width (potentially wasteful)
       or by tying the center state of the sil model to the emitting
       state of the sp model and adding a transitions from 2->4,
       3->2, 4->3 and 4->2 in the sil model.  This reduces its
       minimum duration to 2 frames but more importantly allows
       it to circulate between states.  See Step 7 of the Tutorial
       in the HTKBook.


Conclusions
===========

This has given some basic ideas for how to generate RM systems.
These results can definitely be improved upon - things to try
include.
i)   Cepstral means normalisation (just set TARGETKIND = MFCC_D_A_E_Z)
ii)  Improved model architectures (particularly silence and stops)
iii) Multiple pronunciations
iv)  Further smoothing/tying, particularly tied-mixture systems - try HSmooth
  .... the sky's the limit with HTK


Appendix. Recognition Results
=============================

Recognition results based on models created at various stages of
training are presented here.  The scores are generated by HResults and
are appended to the test LOG files by htestrm.  The path names of the
LOG files given here correspond to the naming convention used in the
recipe.

The SENT scores indicate correctness and accuracy at the sentence
level, while the WORD scores indicate correctness and accuracy at the
word level.  These scores should serve as a guideline for initial
experiments with the RM Toolkit.
Note that at low system complexities the results may be slightly
worse the the HTK_V1.5 RM demo since function word dependent models
are not used in this system.  However the improved capabilities of
HTK_V2.0 (particularly cross-word triphones) lead to better
performance in the end.


Single Mixture Monophone Models
  Rec : R1/hmm4/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=29.33 [H=88, S=212, N=300]
WORD: %Corr=78.33, Acc=76.02 [H=2006, D=136, S=419, I=59, N=2561]
===================================================================

Two-Mixture Monophone Models
  Rec : R2/hmm14/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=39.00 [H=117, S=183, N=300]
WORD: %Corr=84.38, Acc=83.01 [H=2161, D=116, S=284, I=35, N=2561]
===================================================================

Three-Mixture Monophone Models
  Rec : R2/hmm24/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=46.67 [H=140, S=160, N=300]
WORD: %Corr=88.48, Acc=87.50 [H=2266, D=86, S=209, I=25, N=2561]
===================================================================

Four-Mixture Monophone Models
  Rec : R2/hmm34/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=51.67 [H=155, S=145, N=300]
WORD: %Corr=90.00, Acc=89.34 [H=2305, D=87, S=169, I=17, N=2561]
===================================================================

Tied-mixture Monophone Models
  Rec : R3/hmm4/test_wp_feb89.3/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=42.00 [H=126, S=174, N=300]
WORD: %Corr=86.10, Acc=84.62 [H=2205, D=83, S=273, I=38, N=2561]
===================================================================

Discrete-density Monophone Models
  Rec : R4/hmm7/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=41.00 [H=123, S=177, N=300]
WORD: %Corr=85.75, Acc=83.52 [H=2196, D=72, S=293, I=57, N=2561]
===================================================================

Single Mixture, Word-Internal Triphones
  Rec : R5/hmm2/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=52.33 [H=157, S=143, N=300]
WORD: %Corr=90.90, Acc=88.60 [H=2328, D=66, S=167, I=59, N=2561]
===================================================================

Unadapted Single Mixture, Word-Internal Triphones
  Rec : R5/hmm2/dms0_tst/dms0_tst.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=45.92 [H=45, S=53, N=98]
WORD: %Corr=88.36, Acc=85.81 [H=691, D=26, S=65, I=20, N=782]
===================================================================

Adapted Single Mixture, Word-Internal Triphones
  Rec : R5/hmm3/dms0_tst/dms0_tst.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=66.33 [H=65, S=33, N=98]
WORD: %Corr=94.64, Acc=92.85 [H=741, D=10, S=32, I=14, N=783]
===================================================================

Tied-mixture, Word-Internal Triphone Models
  Rec : R6/hmm2/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=55.00 [H=165, S=135, N=300]
WORD: %Corr=90.55, Acc=89.50 [H=2319, D=73, S=169, I=27, N=2561]
===================================================================

Tied-mixture, Smoothed Word-Internal Triphone Models
  Rec : R6/hmm3/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=62.00 [H=186, S=114, N=300]
WORD: %Corr=92.50, Acc=91.84 [H=2369, D=60, S=132, I=17, N=2561]
===================================================================

Single Mixture, Word-Internal, State-Clustered Triphones
  Rec : R8/hmm4/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=54.67 [H=164, S=136, N=300]
WORD: %Corr=91.64, Acc=90.90 [H=2347, D=78, S=136, I=19, N=2561]
===================================================================

Two-Mixture, Word-Internal, State-Clustered Triphones
  Rec : R8/hmm14/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=67.00 [H=201, S=99, N=300]
WORD: %Corr=94.30, Acc=93.87 [H=2415, D=57, S=89, I=11, N=2561]
===================================================================

Three-Mixture, Word-Internal, State-Clustered Triphones
  Rec : R8/hmm24/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=67.67 [H=203, S=97, N=300]
WORD: %Corr=94.81, Acc=94.49 [H=2428, D=49, S=84, I=8, N=2561]
===================================================================

Four-Mixture, Word-Internal, State-Clustered Triphones
  Rec : R8/hmm34/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=69.00 [H=207, S=93, N=300]
WORD: %Corr=95.12, Acc=94.81 [H=2436, D=47, S=78, I=8, N=2561]
===================================================================

Five-Mixture, Word-Internal, State-Clustered Triphones
  Rec : R8/hmm44/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=72.33 [H=217, S=83, N=300]
WORD: %Corr=95.67, Acc=95.35 [H=2450, D=39, S=72, I=8, N=2561]
===================================================================

Six-Mixture, Word-Internal, State-Clustered Triphones
  Rec : R8/hmm54/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=73.00 [H=219, S=81, N=300]
WORD: %Corr=95.78, Acc=95.47 [H=2453, D=43, S=65, I=8, N=2561]
===================================================================

Single Mixture, Cross-Word, State-Clustered Triphones
  Rec : R9/hmm4/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=66.22 [H=198, S=101, N=299]
WORD: %Corr=94.53, Acc=92.96 [H=2418, D=31, S=109, I=40, N=2558]
===================================================================

Six-Mixture, Cross-Word, State-Clustered Triphones
  Rec : R9/hmm54/test_wp_feb89.1/wp_feb89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=81.33 [H=244, S=56, N=300]
WORD: %Corr=97.27, Acc=96.95 [H=2491, D=14, S=56, I=8, N=2561]
===================================================================

  Rec : R9/hmm54/test_wp_oct89.1/wp_oct89.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=76.67 [H=230, S=70, N=300]
WORD: %Corr=96.35, Acc=95.75 [H=2586, D=21, S=77, I=16, N=2684]
===================================================================

  Rec : hmm54/test_wp_feb91.2/wp_feb91.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=81.67 [H=245, S=55, N=300]
WORD: %Corr=97.30, Acc=96.82 [H=2417, D=19, S=48, I=12, N=2484]
===================================================================

  Rec : hmm54/test_wp_sep92.1/wp_sep92.mlf
------------------------ Overall Results --------------------------
SENT: %Correct=67.33 [H=202, S=98, N=300]
WORD: %Corr=94.53, Acc=93.51 [H=2419, D=28, S=112, I=26, N=2559]
===================================================================
