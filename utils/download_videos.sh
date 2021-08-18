cd ../data/DALY
mkdir -p DALY_videos
cd DALY_videos

for filename in `cat ../video_ids.txt`; do
	if [ -f "$filename.mp4" ]; then
		echo "Video $filename exists. Skipping."
	else
		echo "Downloading video $filename.mp4..."
		wget --user user --password password http://pascal.inrialpes.fr/data2/daly_cache/cache/$filename.mp4
	fi
done

