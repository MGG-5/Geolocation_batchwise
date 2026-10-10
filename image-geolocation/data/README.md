# extra data (OSV-5M)

the 60k extra training images, in two batches of 30k. each folder has the image list (`metadata.csv`) and a script that downloads exactly those images from huggingface.

```
pip install huggingface_hub
cd osv5m_batch_a
python fetch_osv5m_subset.py
```

same thing for `osv5m_batch_b`. each batch is about a 7.6 GB download and ends up around 1.5 GB.

the data is from [OSV-5M](https://huggingface.co/datasets/osv5m/osv5m), licensed CC-BY-SA 4.0.
