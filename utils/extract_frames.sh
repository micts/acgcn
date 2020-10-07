input_dir=${1}
output_dir=${2}

cd $input_dir

for filename in *; do
  echo "Processing $filename file..."
  mkdir $output_dir/$filename
  ffmpeg -i $filename $output_dir/$filename/frame%06d.jpg -hide_banner

done
