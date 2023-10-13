# Ensamble-BERT-KBC

Large pre-trained language models are powerful tools for various Natural Language Processing (NLP) tasks due to their ability to
encode a wide range of linguistic and factual knowledge. This potential
has spurred interest in their use for constructing knowledge graphs, a
focus exemplified by the LM-Knowledge Base Construction (LM-KBC)
challenge. The challenge emphasizes populating knowledge graphs from
pre-trained models such as BERT and GPT-3 using provided subjectrelation pairs. Our contribution leverages an ensemble of BERT models,
each trained on a different dataset, with the intent to broaden the coverage of factual knowledge. We explore the potential advantages of domainspecific pre-training in a context of limited task-specific training data.
Furthermore, by using models that generate objects with higher confidence, we facilitate object selection with a universal threshold, simplifying the KBC pipeline. We have demonstrated models that outperform
the baseline BERT model for certain relations, and have constructed an
ensemble of these high-performing models. The results show that the ensemble model outperforms the baseline model using a common threshold
for all relations, relying solely on the provided LM-KBC data.
