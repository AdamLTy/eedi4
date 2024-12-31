#!/bin/bash
cp ../requirements.txt requirements.txt
TAG="kaggle:eedi_nvcr"
docker build -f Dockerfile -t $TAG .
rm requirements.txt
