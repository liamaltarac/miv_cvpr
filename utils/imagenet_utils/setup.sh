for file in /c/ILSVRC2012_img_val/*.JPEG; do 
    read line;
    d=`basename $file`
    mkdir -p /c/ILSVRC2012_img_val/${line}
    mv -v "${file}" "/c/ILSVRC2012_img_val/${line}"; 
 done < Desktop/filter_exploration/comparison_plots/imagenet_utils/ILSVRC2012_validation_ground_truth_synset.txt