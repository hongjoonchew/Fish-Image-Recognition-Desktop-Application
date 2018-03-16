# Bash script to process folders of images
# Using the augDataSetBB.py augmentation script

filenames=(1-50 1-75_176-200 1-100 51-100 76-150 101-150 151-200)
batchid=0
index=0

while [ $batchid -lt 100 ] # Creates 100 batches
do
	python ./augDataSetBB.py ../PositiveDataSet/${filenames[index]} output ../PositiveDataSet/annotations/${filenames[index]}.json;
	cd output;
	zip batch$batchid *;           # Zip images in output folder
	mv batch$batchid.zip ../out;   # Move output zip to out folder
	rm *;                          # Delete images in output folder to prep for new batch
	cd ..;

	index=$((index+1))
	batchid=$((batchid+1))

	index=$((index%7)) 
done
