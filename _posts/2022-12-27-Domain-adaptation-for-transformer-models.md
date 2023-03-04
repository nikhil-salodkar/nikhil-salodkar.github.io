---
title: Domain adaptation in transformer models
tags: [Transformer Architectures, NLP]
style: fill
color: primary
comments: true
description: Description of several domain adaptation techniques
---

Transformer models have proven to be quite performative and useful when correctly pre-trained on large text corpus 
and fine-tuned with small to moderately sized supervised datasets to solve downstream NLP tasks. Much of the research
in pre-training of Language models, and testing performance is done using general text corpus like Wikipedia, News
articles, Reddit comments and mixture of such types of text corpus sources. The main reason is easier availability 
of such resources and trying to acquire as much text and as varied text as possible for research purposes. So, many
of the popular pre-trained models turned out to be pre-trained using general purpose text which cannot necessarily
be categorized as a particular domain.

But, in real life and most of the production scenarios applications are made to handle a particular domain of text
data. For example, application to summarize news articles, application to perform search over scientific research
documents, chatbot built to answer education related queries and so on. Thus, these specialized application focus
on particular domains of their own and, it begs the question, whether using a general purpose trained language
model to train for such specialized domain is the best thing to do. This question led to the research community in
seeking the area of domain adaptation for language models.

In this post we will understand what domain adaptation is, its benefits and how it can be done.

# What is Domain Adaptation and Why it is important?
Domain adaptation can be defined as any training technique which can help a pre-trained Language model to better
adapt itself to solve tasks in a specialized NLP text domain.

This can be important because in real world there are many NLP domains with common tasks which can benefit if 
the language model is adapted suitably. Some common NLP domains and popular tasks are as given below:
- Medical and/or Pharmaceutical domain : Popular tasks and applications in this domain include
  - Entity detection and Linking : Detecting special pharmaceutical and medical entities in a text like entities
    pertaining to diseases, medicine, proteins, symptoms, side effects etc.
  - Specialized chatbots for answering patient queries and return correct search results
  - Research paper recommendation : What research to read next pertaining to research area of interest
  - Extractive text summarization : Automatically highlighting important sentences in medical research papers etc
- General News domain :
  - News article summarization
  - Article recommendation based on readers interest and past history
  - Clustering news articles and finding similar or correlated news
- Product sales and services :
  - Sentiment analysis on reviews of purchased products like retail products or even watched movies or games played
  - Customer support service assistance for faster solution resolution
  - Intelligent IVR systems

Many NLP applications are focused on a particular industry and may also have focused subdomain in that industry,
like in Pharma industry language around clinical trials might be different to drug discovery research and very
different compared to clinical Electronic Medical records language. There could be difference in vocabulary,
sentence structure, short forms, language formality and so on. Thus, in most real world cases it may become
necessary to specialize language model to get better performance on interested tasks.

To create such specialized models one approach is to train the transformer based language models directly using
relevant domain text data. This has been done on some popular models like **BioMegatron, Sci-Bert, BioBERT** etc which
were pre-trained directly on specialized text corpus like Pubmed extracts and other scientific journals. But, in
most cases for specialized domains there either would be dearth in quantity of text necessary to train a full-fledged
language model, or it would be too computationally expensive and time-consuming.

Thus, it becomes prudent that we find useful training methods that could help in adaptation of existing Language models
more efficiently.

# Domain adaptation strategies
Let's review some domain adaptation strategies that we can follow and discuss some on some research papers
proposing these ideas.

- **Adapting domain by continuing the pre-training using in-domain dataset**:
- **Updating tokenization strategy to include more in-domain vocabulary**: