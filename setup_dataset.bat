@echo off

REM Download LIAR Dataset as specified in README
echo Downloading LIAR dataset...

REM Create dataset directory if it doesn't exist
if not exist liar_dataset mkdir liar_dataset

REM Download dataset files from the original LIAR dataset repository
curl -o liar_dataset/train.tsv https://raw.githubusercontent.com/williamyang1991/LIAR-dataset/master/dataset/train.tsv
curl -o liar_dataset/valid.tsv https://raw.githubusercontent.com/williamyang1991/LIAR-dataset/master/dataset/valid.tsv
curl -o liar_dataset/test.tsv https://raw.githubusercontent.com/williamyang1991/LIAR-dataset/master/dataset/test.tsv

echo Dataset download complete!
echo Files created:
dir liar_dataset