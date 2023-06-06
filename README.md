# README

## Exploring the Potential of Transformer Models as a Novel Approach to Biological Knowledge Graph Construction

## Abstract
Biological data is littered with complex terms and relationships. One way to represent these relationships is through Knowledge Graphs. Over the past half decade, transformer models have revolutionised many NLP tasks. BERT models have been increasingly employed in other academic spheres for KG creation. Throughout this project, we explored the potential for transformer models within the field of biology, evaluating their ability to generate high quality triples through qualitatively (through visualisation) and quantitatively (through training predictive models).

## Research Question
Our research question is as follows: To what extent can transformer models be utilised to generate high quality triples from a corpus of biological abstracts.

## Dataset
We use a curated dataset made from programmatically downloading PubMed article abstracts. Further details are contained in ```project_notebook.ipynb```.

## Documentation
This repository contains two notebooks, one focused on triplet extraction and NLP-based techniques, and the other on visualisation methods used throughout the project. It also contains a data folder, in which all relevant files created over the course of the project have been stored. This includes CSV files containing triplets at each step of the process, the corpus itself, and predictor model evaluation reports. 

As well as this, it contains a series of generated knowledge graphs, processed in [Gephi](https://gephi.org) and exported as PDFs for readers to explore.
