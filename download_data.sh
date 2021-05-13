#!/bin/bash

# https://www.matthuisman.nz/2019/01/download-google-drive-files-wget-curl.html

export fileid=1LSiIqQcgqoBTqrJ8MPIcWWcAM8O2QGnx
export filename=data.tar.xz

echo "Downloading..."
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- |
  sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' >confirm.txt

wget --load-cookies cookies.txt -O $filename \
  'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)

echo "Extracting..."
tar -xf $filename

echo "Cleaning up..."
rm -f confirm.txt cookies.txt $filename

echo "Done!"
