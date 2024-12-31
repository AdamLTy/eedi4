#!/bin/bash
TAG="kaggle:eedi_vllm"
docker build -f Dockerfile -t $TAG .
