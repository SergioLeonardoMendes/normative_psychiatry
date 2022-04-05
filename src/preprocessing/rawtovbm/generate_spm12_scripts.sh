# List with all .nii files to be VBM processed. The file paths must be relative to the spm12 docker container.
ALLFILES=all_nii_files.txt
# Number of lines to split file list in multiple jobs (PART 1 only)
NSPLIT=1123
# template filenames to replace strings
JOBTEMPLATE1=spm12_template_part1.m
JOBTEMPLATE2=spm12_template_part2.m
JOBTEMPLATESTART=spm12_template_start.m
# Processing variables (there is no need to change) 
JOBPREFIX1=spm12_job1 
JOBPREFIX2=spm12_job2 
FILES1=files_1.txt
FILES2RC1=files_2_rc1.txt
FILES2RC2=files_2_rc2.txt
FILES3URC1=files_3_u_rc1.txt
FILES3C1=files_3_c1.txt
FILES3C2=files_3_c2.txt
FILES3C3=files_3_c3.txt


### Attention needed: change path from Template_6.nii in "spm12_job_run.m"
### The path should be the first from the file ALLFILES


echo "##### Generating files PART 1 #####";
split -l $NSPLIT -d $ALLFILES;
for FILEPART in `ls -1 x0*`; do
    rm -f $FILES1;
    for LINEPATH in `cat $FILEPART`; do
	DNAME=`dirname "$LINEPATH"`
	FNAME=`basename "$LINEPATH"`
	echo "'$DNAME/${FNAME},1'" >> $FILES1;
    done
    
    JOBFILENAME=$JOBPREFIX1"_$FILEPART".m
    JOBFILENAMESTART="start_"$JOBFILENAME

    cp $JOBTEMPLATE1 $JOBFILENAME
    cp $JOBTEMPLATESTART "start_"$JOBFILENAME
    sed -e "/XXX_1_XXX/ {" -e "r $FILES1" -e "d" -e "}" -i $JOBFILENAME
    sed -i "s/XXX_JOBFILE_XXX/$JOBFILENAME/" $JOBFILENAMESTART
done

echo "##### Generating files PART 2 #####";
rm -f $FILES1;
rm -f $FILES2RC1;
rm -f $FILES2RC2;
rm -f $FILES3URC1;
rm -f $FILES3C1;
rm -f $FILES3C2;
rm -f $FILES3C3;
for LINEPATH in `cat $ALLFILES`; do
	DNAME=`dirname "$LINEPATH"`
	FNAME=`basename "$LINEPATH"`

	echo "'$DNAME/${FNAME},1'" >> $FILES1;
	echo "'$DNAME/rc1${FNAME},1'" >> $FILES2RC1;
	echo "'$DNAME/rc2${FNAME},1'" >> $FILES2RC2;

	PARTNAME=`echo ${FNAME} | awk -F. '{print $1}'`
	echo "'$DNAME/u_rc1${PARTNAME}_Template.nii'" >> $FILES3URC1;

	echo "'$DNAME/c1${FNAME}'" >> $FILES3C1;
	echo "'$DNAME/c2${FNAME}'" >> $FILES3C2;
	echo "'$DNAME/c3${FNAME}'" >> $FILES3C3;
done

JOBFILENAME=$JOBPREFIX2.m
JOBFILENAMESTART="start_"$JOBFILENAME
cp $JOBTEMPLATE2 $JOBFILENAME
cp $JOBTEMPLATESTART $JOBFILENAMESTART
sed -e "/XXX_2RC1_XXX/ {" -e "r $FILES2RC1" -e "d" -e "}" -i $JOBFILENAME
sed -e "/XXX_2RC2_XXX/ {" -e "r $FILES2RC2" -e "d" -e "}" -i $JOBFILENAME
sed -e "/XXX_3URC1_XXX/ {" -e "r $FILES3URC1" -e "d" -e "}" -i $JOBFILENAME
sed -e "/XXX_3C1_XXX/ {" -e "r $FILES3C1" -e "d" -e "}" -i $JOBFILENAME
sed -e "/XXX_3C2_XXX/ {" -e "r $FILES3C2" -e "d" -e "}" -i $JOBFILENAME
sed -e "/XXX_3C3_XXX/ {" -e "r $FILES3C3" -e "d" -e "}" -i $JOBFILENAME
sed -i "s/XXX_JOBFILE_XXX/$JOBFILENAME/" $JOBFILENAMESTART

echo "##### Cleaning temporary files #####";
rm -f $FILES1;
rm -f $FILES2RC1;
rm -f $FILES2RC2;
rm -f $FILES3URC1;
rm -f $FILES3C1;
rm -f $FILES3C2;
rm -f $FILES3C3;
rm -f x0*;
