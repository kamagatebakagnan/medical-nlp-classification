# Medical NLP Classification

Classification automatique de transcriptions médicales par spécialité.
Dataset : [Medical Transcriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
(4 610 documents · 22 spécialités médicales)

## Problématique
Trier automatiquement des notes médicales par spécialité
(chirurgie, cardiologie, neurologie...) pour aider les
hôpitaux à organiser leurs dossiers patients.

## Pipeline
```
Texte brut → Nettoyage → TF-IDF / Word2Vec / BioBERT → Classification
```

## Résultats

| Approche | F1-macro | F1-weighted |
|----------|----------|-------------|
| TF-IDF + LogisticRegression | **0.386** | 0.306 |
| Word2Vec + LogisticRegression | 0.368 | 0.294 |
| DistilBERT (3 epochs) | 0.177 | 0.305 |
| BioBERT (5 epochs) | 0.253 | 0.304 |

Meilleure approche : **TF-IDF + LogisticRegression**

## Insights clés

- TF-IDF surpasse BERT sur ce dataset car les termes médicaux
  rares sont très discriminants et le dataset est petit (~167 docs/classe)
- BioBERT (pré-entraîné sur PubMed) surpasse DistilBERT de +43%
  → l'importance du domaine de pré-entraînement
- Overfitting BioBERT détecté dès l'epoch 4 → meilleur checkpoint = epoch 3
- Spécialités bien classifiées : Psychiatry (F1=0.69), Ophthalmology (0.64)
- Spécialités difficiles : Surgery (0.12), General Medicine (0.13)
  → vocabulaire trop générique

## Choix techniques
- Métrique : F1-macro (multiclasse déséquilibré)
- Nettoyage : lowercase, regex, suppression stopwords
- TF-IDF : ngram_range=(1,2), sublinear_tf=True, max_features=10000
- Word2Vec : Skip-gram, vector_size=100, window=5
- BERT : fine-tuning HuggingFace sur GPU T4 (Google Colab)

## Structure
```
medical-nlp-classification/
├── notebooks/
│   ├── 01_tfidf_word2vec.ipynb      ← TF-IDF + Word2Vec sur Kaggle
│   └── 02_biobert_finetuning.ipynb  ← BioBERT sur Google Colab
├── .gitignore
├── LICENSE
└── README.md
```

## Installation
```bash
git clone https://github.com/kamagatebakagnan/medical-nlp-classification.git
cd medical-nlp-classification
pip install pandas numpy scikit-learn gensim transformers torch
jupyter notebook notebooks/01_tfidf_word2vec.ipynb
```
```

Commit avec le message :
```
docs: README complet avec résultats et insights médicaux
