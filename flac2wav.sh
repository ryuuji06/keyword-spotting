# Convert all .flac files within this folder to .wav files

# to execute in windows
# powershell ./flac2wav.sh

# uses avconv program from libav
# http://builds.libav.org/windows/nightly-gpl/

find . -iname "*.flac" | wc

for flacfile in `find . -iname "*.flac"`
do
    avconv -y -f flac -i $flacfile -ab 64k -ac 1 -ar 16000 -f wav "${flacfile%.*}.wav"
	rm "${flacfile%.*}.flac"
done
