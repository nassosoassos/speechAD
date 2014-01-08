


use strict;

my $Task='';
my $RATSID='';
my $AudioPath='';
my $AudioFile='';
my $Channel='';
my $AnnotFile='';
my $SampleCounter=0;
my %TestHash=();
my $InputFile='';
my $OutputFile='';
my $Help=0;
my $AnswersFile='';
my $ResFile='';

my $PerlProgramName=$0;

# Input arguments
while(@ARGV){
    $_=shift;
    /^-p|--path$/  && ($AudioPath=shift,next);  
    /^-i|--input$/ && ($InputFile=shift,next);
    /^-o|--output$/ && ($OutputFile=shift,next);
    /^-a|--answers-file$/ && ($AnswersFile=shift,next);
    /^-t|--task$/  && ($Task=shift,next);
    /^-r|--rats-id$/ && ($RATSID=shift,next);
    /^-h|--help$/    && ($Help=1,next);
    die("Wrong arguments used.\nFor help run: $PerlProgramName --help ");
}



#Help menu
if ($Help==1){
    print STDERR "Usage example: perl $PerlProgramName --input <INPUTFILE> --output <OUTPUTFILE> -t SAD -p /Audio/path/\n";
    print STDERR "Usage example: cat <INPUTFILE> | perl $PerlProgramName -t SAD -p /Audio/path/ > <OUTPUTFILE>\n\n";
    print STDERR  "\t------------------------------------------------------\n";
    print STDERR "COMMAND LINE OPTIONS\n";
    print STDERR "\tOPTIONS       ALTERNATIVE               EXPLANATION\n";
    print STDERR "\t-------       -----------               -----------\n";
    print STDERR "\t-i            --input                   Input file \n";
    print STDERR "\t-o            --output                  Output file\n";
    print STDERR "\t-a            --answers-file            Answers file\n";
    print STDERR "\t-p            --path                    Audio Files path\n";
    print STDERR "\t-t            --task                    Specify the task, e.g. SAD\n";
    print STDERR "\t-r            --rats-id                 Specify the rats-id, e.g. MYID\n";
    print STDERR "\t-h            --help                    This help menu\n\n\n";
    print STDERR "Input file is a three columns file. The first column has the audio file, the second column has the channel and the third column has the annotation file.\n\n";
    print STDERR "Output is RATS software test definition file.\n\n";
    exit(1);
}

# Mandatory agruments
die("Please specify the audio path\nFor help run: $PerlProgramName --help") if ($AudioPath  eq '');
die("Please specify the correct task\nFor help run: $PerlProgramName --help") if($Task ne 'SAD');
$RATSID="DefaultID" if ($RATSID=='');

open(STDIN,$InputFile) if ($InputFile ne '');
open(STDOUT,">$OutputFile") if ($OutputFile ne '');
open(ALLRESULTSFILE,">$AnswersFile") if ($AnswersFile ne '');

print STDOUT "<RATSTestSet id=\"$RATSID\" audio=\"$AudioPath\" task=\"$Task\">\n";


# Repeat for all lines
while (<STDIN>){
    # Extract the audio file, channel and annotation file for each line
    $_=~ m/(.*?)\s+(.*?)\s+(.*?)\s+(.*?)\n/;
    $AudioFile=$1;
    $Channel=$2;
    $AnnotFile=$3;
    $ResFile=$4;
    $SampleCounter++;
    die("Cannot open annotations file.") if(!open(ANNOTFILE,$AnnotFile));
    $AnnotFile =~ s/\//_/g;
    $AnnotFile =~ s/\./_/g;
    $AnnotFile =~ s/\-/_/g;
    die("Cannot open answers file.") if(!open(RESULTSFILE,$ResFile));
    print STDOUT "\t<SAMPLE id=\"Sample$SampleCounter\" file=\"$1\">\n";
    # Read annotation segments
    while(<ANNOTFILE>){
        $_=~ m/(.*?)\s+(.*?)\n/;
        print STDOUT "\t\t<SEGMENT start=\"",$1,"\" end=\"",$2,"\" />\n";
    }
    close(ANNOTFILE);
    
    while(<RESULTSFILE>){
        $_=~ m/(.*?)\s+(.*?)\n/;
        print ALLRESULTSFILE "$OutputFile\t$RATSID\t$Channel\t$Task\tSample$SampleCounter\t",$1,"\t",$2,"\n";
    }    

    # Store tests for each channel to print after reading all the samples
    $TestHash{$Channel}="\t<TEST id=\"$Channel\">\n" if (! exists $TestHash{$Channel});
    $TestHash{$Channel} = $TestHash{$Channel}. "\t\t<SAMPLE ref=\"Sample$SampleCounter\" />\n";
    
    print STDOUT "\t</SAMPLE>\n";
    
}

# Print all test tags
while( my ($keys, $values) = each(%TestHash)){
    print STDOUT "$values\t</TEST>\n";
}


print STDOUT "</RATSTestSet>"
