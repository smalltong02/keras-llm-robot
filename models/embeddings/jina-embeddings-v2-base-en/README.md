---
tags:
  - finetuner
  - mteb
  - sentence-transformers
  - feature-extraction
  - sentence-similarity
  - alibi
datasets:
  - allenai/c4
language: en
inference: false
license: apache-2.0
model-index:
- name: jina-embedding-b-en-v2
  results:
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_counterfactual
      name: MTEB AmazonCounterfactualClassification (en)
      config: en
      split: test
      revision: e8379541af4e31359cca9fbcf4b00f2671dba205
    metrics:
    - type: accuracy
      value: 74.73134328358209
    - type: ap
      value: 37.765427081831035
    - type: f1
      value: 68.79367444339518
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_polarity
      name: MTEB AmazonPolarityClassification
      config: default
      split: test
      revision: e2d317d38cd51312af73b3d32a06d1a08b442046
    metrics:
    - type: accuracy
      value: 88.544275
    - type: ap
      value: 84.61328675662887
    - type: f1
      value: 88.51879035862375
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (en)
      config: en
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 45.263999999999996
    - type: f1
      value: 43.778759656699435
  - task:
      type: Retrieval
    dataset:
      type: arguana
      name: MTEB ArguAna
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 21.693
    - type: map_at_10
      value: 35.487
    - type: map_at_100
      value: 36.862
    - type: map_at_1000
      value: 36.872
    - type: map_at_3
      value: 30.049999999999997
    - type: map_at_5
      value: 32.966
    - type: mrr_at_1
      value: 21.977
    - type: mrr_at_10
      value: 35.565999999999995
    - type: mrr_at_100
      value: 36.948
    - type: mrr_at_1000
      value: 36.958
    - type: mrr_at_3
      value: 30.121
    - type: mrr_at_5
      value: 33.051
    - type: ndcg_at_1
      value: 21.693
    - type: ndcg_at_10
      value: 44.181
    - type: ndcg_at_100
      value: 49.982
    - type: ndcg_at_1000
      value: 50.233000000000004
    - type: ndcg_at_3
      value: 32.830999999999996
    - type: ndcg_at_5
      value: 38.080000000000005
    - type: precision_at_1
      value: 21.693
    - type: precision_at_10
      value: 7.248
    - type: precision_at_100
      value: 0.9769999999999999
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 13.632
    - type: precision_at_5
      value: 10.725
    - type: recall_at_1
      value: 21.693
    - type: recall_at_10
      value: 72.475
    - type: recall_at_100
      value: 97.653
    - type: recall_at_1000
      value: 99.57300000000001
    - type: recall_at_3
      value: 40.896
    - type: recall_at_5
      value: 53.627
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-p2p
      name: MTEB ArxivClusteringP2P
      config: default
      split: test
      revision: a122ad7f3f0291bf49cc6f4d32aa80929df69d5d
    metrics:
    - type: v_measure
      value: 45.39242428696777
  - task:
      type: Clustering
    dataset:
      type: mteb/arxiv-clustering-s2s
      name: MTEB ArxivClusteringS2S
      config: default
      split: test
      revision: f910caf1a6075f7329cdf8c1a6135696f37dbd53
    metrics:
    - type: v_measure
      value: 36.675626784714
  - task:
      type: Reranking
    dataset:
      type: mteb/askubuntudupquestions-reranking
      name: MTEB AskUbuntuDupQuestions
      config: default
      split: test
      revision: 2000358ca161889fa9c082cb41daa8dcfb161a54
    metrics:
    - type: map
      value: 62.247725694904034
    - type: mrr
      value: 74.91359978894604
  - task:
      type: STS
    dataset:
      type: mteb/biosses-sts
      name: MTEB BIOSSES
      config: default
      split: test
      revision: d3fb88f8f02e40887cd149695127462bbcf29b4a
    metrics:
    - type: cos_sim_pearson
      value: 82.68003802970496
    - type: cos_sim_spearman
      value: 81.23438110096286
    - type: euclidean_pearson
      value: 81.87462986142582
    - type: euclidean_spearman
      value: 81.23438110096286
    - type: manhattan_pearson
      value: 81.61162566600755
    - type: manhattan_spearman
      value: 81.11329400456184
  - task:
      type: Classification
    dataset:
      type: mteb/banking77
      name: MTEB Banking77Classification
      config: default
      split: test
      revision: 0fd18e25b25c072e09e0d92ab615fda904d66300
    metrics:
    - type: accuracy
      value: 84.01298701298701
    - type: f1
      value: 83.31690714969382
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-p2p
      name: MTEB BiorxivClusteringP2P
      config: default
      split: test
      revision: 65b79d1d13f80053f67aca9498d9402c2d9f1f40
    metrics:
    - type: v_measure
      value: 37.050108150972086
  - task:
      type: Clustering
    dataset:
      type: mteb/biorxiv-clustering-s2s
      name: MTEB BiorxivClusteringS2S
      config: default
      split: test
      revision: 258694dd0231531bc1fd9de6ceb52a0853c6d908
    metrics:
    - type: v_measure
      value: 30.15731442819715
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackAndroidRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 31.391999999999996
    - type: map_at_10
      value: 42.597
    - type: map_at_100
      value: 44.07
    - type: map_at_1000
      value: 44.198
    - type: map_at_3
      value: 38.957
    - type: map_at_5
      value: 40.961
    - type: mrr_at_1
      value: 37.196
    - type: mrr_at_10
      value: 48.152
    - type: mrr_at_100
      value: 48.928
    - type: mrr_at_1000
      value: 48.964999999999996
    - type: mrr_at_3
      value: 45.446
    - type: mrr_at_5
      value: 47.205999999999996
    - type: ndcg_at_1
      value: 37.196
    - type: ndcg_at_10
      value: 49.089
    - type: ndcg_at_100
      value: 54.471000000000004
    - type: ndcg_at_1000
      value: 56.385
    - type: ndcg_at_3
      value: 43.699
    - type: ndcg_at_5
      value: 46.22
    - type: precision_at_1
      value: 37.196
    - type: precision_at_10
      value: 9.313
    - type: precision_at_100
      value: 1.478
    - type: precision_at_1000
      value: 0.198
    - type: precision_at_3
      value: 20.839
    - type: precision_at_5
      value: 14.936
    - type: recall_at_1
      value: 31.391999999999996
    - type: recall_at_10
      value: 61.876
    - type: recall_at_100
      value: 84.214
    - type: recall_at_1000
      value: 95.985
    - type: recall_at_3
      value: 46.6
    - type: recall_at_5
      value: 53.588
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackEnglishRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 29.083
    - type: map_at_10
      value: 38.812999999999995
    - type: map_at_100
      value: 40.053
    - type: map_at_1000
      value: 40.188
    - type: map_at_3
      value: 36.111
    - type: map_at_5
      value: 37.519000000000005
    - type: mrr_at_1
      value: 36.497
    - type: mrr_at_10
      value: 44.85
    - type: mrr_at_100
      value: 45.546
    - type: mrr_at_1000
      value: 45.593
    - type: mrr_at_3
      value: 42.686
    - type: mrr_at_5
      value: 43.909
    - type: ndcg_at_1
      value: 36.497
    - type: ndcg_at_10
      value: 44.443
    - type: ndcg_at_100
      value: 48.979
    - type: ndcg_at_1000
      value: 51.154999999999994
    - type: ndcg_at_3
      value: 40.660000000000004
    - type: ndcg_at_5
      value: 42.193000000000005
    - type: precision_at_1
      value: 36.497
    - type: precision_at_10
      value: 8.433
    - type: precision_at_100
      value: 1.369
    - type: precision_at_1000
      value: 0.185
    - type: precision_at_3
      value: 19.894000000000002
    - type: precision_at_5
      value: 13.873
    - type: recall_at_1
      value: 29.083
    - type: recall_at_10
      value: 54.313
    - type: recall_at_100
      value: 73.792
    - type: recall_at_1000
      value: 87.629
    - type: recall_at_3
      value: 42.257
    - type: recall_at_5
      value: 47.066
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackGamingRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 38.556000000000004
    - type: map_at_10
      value: 50.698
    - type: map_at_100
      value: 51.705
    - type: map_at_1000
      value: 51.768
    - type: map_at_3
      value: 47.848
    - type: map_at_5
      value: 49.358000000000004
    - type: mrr_at_1
      value: 43.95
    - type: mrr_at_10
      value: 54.191
    - type: mrr_at_100
      value: 54.852999999999994
    - type: mrr_at_1000
      value: 54.885
    - type: mrr_at_3
      value: 51.954
    - type: mrr_at_5
      value: 53.13
    - type: ndcg_at_1
      value: 43.95
    - type: ndcg_at_10
      value: 56.516
    - type: ndcg_at_100
      value: 60.477000000000004
    - type: ndcg_at_1000
      value: 61.746
    - type: ndcg_at_3
      value: 51.601
    - type: ndcg_at_5
      value: 53.795
    - type: precision_at_1
      value: 43.95
    - type: precision_at_10
      value: 9.009
    - type: precision_at_100
      value: 1.189
    - type: precision_at_1000
      value: 0.135
    - type: precision_at_3
      value: 22.989
    - type: precision_at_5
      value: 15.473
    - type: recall_at_1
      value: 38.556000000000004
    - type: recall_at_10
      value: 70.159
    - type: recall_at_100
      value: 87.132
    - type: recall_at_1000
      value: 96.16
    - type: recall_at_3
      value: 56.906
    - type: recall_at_5
      value: 62.332
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackGisRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 24.238
    - type: map_at_10
      value: 32.5
    - type: map_at_100
      value: 33.637
    - type: map_at_1000
      value: 33.719
    - type: map_at_3
      value: 30.026999999999997
    - type: map_at_5
      value: 31.555
    - type: mrr_at_1
      value: 26.328000000000003
    - type: mrr_at_10
      value: 34.44
    - type: mrr_at_100
      value: 35.455999999999996
    - type: mrr_at_1000
      value: 35.521
    - type: mrr_at_3
      value: 32.034
    - type: mrr_at_5
      value: 33.565
    - type: ndcg_at_1
      value: 26.328000000000003
    - type: ndcg_at_10
      value: 37.202
    - type: ndcg_at_100
      value: 42.728
    - type: ndcg_at_1000
      value: 44.792
    - type: ndcg_at_3
      value: 32.368
    - type: ndcg_at_5
      value: 35.008
    - type: precision_at_1
      value: 26.328000000000003
    - type: precision_at_10
      value: 5.7059999999999995
    - type: precision_at_100
      value: 0.8880000000000001
    - type: precision_at_1000
      value: 0.11100000000000002
    - type: precision_at_3
      value: 13.672
    - type: precision_at_5
      value: 9.74
    - type: recall_at_1
      value: 24.238
    - type: recall_at_10
      value: 49.829
    - type: recall_at_100
      value: 75.21
    - type: recall_at_1000
      value: 90.521
    - type: recall_at_3
      value: 36.867
    - type: recall_at_5
      value: 43.241
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackMathematicaRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 15.378
    - type: map_at_10
      value: 22.817999999999998
    - type: map_at_100
      value: 23.977999999999998
    - type: map_at_1000
      value: 24.108
    - type: map_at_3
      value: 20.719
    - type: map_at_5
      value: 21.889
    - type: mrr_at_1
      value: 19.03
    - type: mrr_at_10
      value: 27.022000000000002
    - type: mrr_at_100
      value: 28.011999999999997
    - type: mrr_at_1000
      value: 28.096
    - type: mrr_at_3
      value: 24.855
    - type: mrr_at_5
      value: 26.029999999999998
    - type: ndcg_at_1
      value: 19.03
    - type: ndcg_at_10
      value: 27.526
    - type: ndcg_at_100
      value: 33.040000000000006
    - type: ndcg_at_1000
      value: 36.187000000000005
    - type: ndcg_at_3
      value: 23.497
    - type: ndcg_at_5
      value: 25.334
    - type: precision_at_1
      value: 19.03
    - type: precision_at_10
      value: 4.963
    - type: precision_at_100
      value: 0.893
    - type: precision_at_1000
      value: 0.13
    - type: precision_at_3
      value: 11.360000000000001
    - type: precision_at_5
      value: 8.134
    - type: recall_at_1
      value: 15.378
    - type: recall_at_10
      value: 38.061
    - type: recall_at_100
      value: 61.754
    - type: recall_at_1000
      value: 84.259
    - type: recall_at_3
      value: 26.788
    - type: recall_at_5
      value: 31.326999999999998
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackPhysicsRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 27.511999999999997
    - type: map_at_10
      value: 37.429
    - type: map_at_100
      value: 38.818000000000005
    - type: map_at_1000
      value: 38.924
    - type: map_at_3
      value: 34.625
    - type: map_at_5
      value: 36.064
    - type: mrr_at_1
      value: 33.300999999999995
    - type: mrr_at_10
      value: 43.036
    - type: mrr_at_100
      value: 43.894
    - type: mrr_at_1000
      value: 43.936
    - type: mrr_at_3
      value: 40.825
    - type: mrr_at_5
      value: 42.028
    - type: ndcg_at_1
      value: 33.300999999999995
    - type: ndcg_at_10
      value: 43.229
    - type: ndcg_at_100
      value: 48.992000000000004
    - type: ndcg_at_1000
      value: 51.02100000000001
    - type: ndcg_at_3
      value: 38.794000000000004
    - type: ndcg_at_5
      value: 40.65
    - type: precision_at_1
      value: 33.300999999999995
    - type: precision_at_10
      value: 7.777000000000001
    - type: precision_at_100
      value: 1.269
    - type: precision_at_1000
      value: 0.163
    - type: precision_at_3
      value: 18.351
    - type: precision_at_5
      value: 12.762
    - type: recall_at_1
      value: 27.511999999999997
    - type: recall_at_10
      value: 54.788000000000004
    - type: recall_at_100
      value: 79.105
    - type: recall_at_1000
      value: 92.49199999999999
    - type: recall_at_3
      value: 41.924
    - type: recall_at_5
      value: 47.026
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackProgrammersRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 24.117
    - type: map_at_10
      value: 33.32
    - type: map_at_100
      value: 34.677
    - type: map_at_1000
      value: 34.78
    - type: map_at_3
      value: 30.233999999999998
    - type: map_at_5
      value: 31.668000000000003
    - type: mrr_at_1
      value: 29.566
    - type: mrr_at_10
      value: 38.244
    - type: mrr_at_100
      value: 39.245000000000005
    - type: mrr_at_1000
      value: 39.296
    - type: mrr_at_3
      value: 35.864000000000004
    - type: mrr_at_5
      value: 36.919999999999995
    - type: ndcg_at_1
      value: 29.566
    - type: ndcg_at_10
      value: 39.127
    - type: ndcg_at_100
      value: 44.989000000000004
    - type: ndcg_at_1000
      value: 47.189
    - type: ndcg_at_3
      value: 34.039
    - type: ndcg_at_5
      value: 35.744
    - type: precision_at_1
      value: 29.566
    - type: precision_at_10
      value: 7.385999999999999
    - type: precision_at_100
      value: 1.204
    - type: precision_at_1000
      value: 0.158
    - type: precision_at_3
      value: 16.286
    - type: precision_at_5
      value: 11.484
    - type: recall_at_1
      value: 24.117
    - type: recall_at_10
      value: 51.559999999999995
    - type: recall_at_100
      value: 77.104
    - type: recall_at_1000
      value: 91.79899999999999
    - type: recall_at_3
      value: 36.82
    - type: recall_at_5
      value: 41.453
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 25.17625
    - type: map_at_10
      value: 34.063916666666664
    - type: map_at_100
      value: 35.255500000000005
    - type: map_at_1000
      value: 35.37275
    - type: map_at_3
      value: 31.351666666666667
    - type: map_at_5
      value: 32.80608333333333
    - type: mrr_at_1
      value: 29.59783333333333
    - type: mrr_at_10
      value: 38.0925
    - type: mrr_at_100
      value: 38.957249999999995
    - type: mrr_at_1000
      value: 39.01608333333333
    - type: mrr_at_3
      value: 35.77625
    - type: mrr_at_5
      value: 37.04991666666667
    - type: ndcg_at_1
      value: 29.59783333333333
    - type: ndcg_at_10
      value: 39.343666666666664
    - type: ndcg_at_100
      value: 44.488249999999994
    - type: ndcg_at_1000
      value: 46.83358333333334
    - type: ndcg_at_3
      value: 34.69708333333333
    - type: ndcg_at_5
      value: 36.75075
    - type: precision_at_1
      value: 29.59783333333333
    - type: precision_at_10
      value: 6.884083333333332
    - type: precision_at_100
      value: 1.114
    - type: precision_at_1000
      value: 0.15108333333333332
    - type: precision_at_3
      value: 15.965250000000003
    - type: precision_at_5
      value: 11.246500000000001
    - type: recall_at_1
      value: 25.17625
    - type: recall_at_10
      value: 51.015999999999984
    - type: recall_at_100
      value: 73.60174999999998
    - type: recall_at_1000
      value: 89.849
    - type: recall_at_3
      value: 37.88399999999999
    - type: recall_at_5
      value: 43.24541666666666
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackStatsRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 24.537
    - type: map_at_10
      value: 31.081999999999997
    - type: map_at_100
      value: 32.042
    - type: map_at_1000
      value: 32.141
    - type: map_at_3
      value: 29.137
    - type: map_at_5
      value: 30.079
    - type: mrr_at_1
      value: 27.454
    - type: mrr_at_10
      value: 33.694
    - type: mrr_at_100
      value: 34.579
    - type: mrr_at_1000
      value: 34.649
    - type: mrr_at_3
      value: 32.004
    - type: mrr_at_5
      value: 32.794000000000004
    - type: ndcg_at_1
      value: 27.454
    - type: ndcg_at_10
      value: 34.915
    - type: ndcg_at_100
      value: 39.641
    - type: ndcg_at_1000
      value: 42.105
    - type: ndcg_at_3
      value: 31.276
    - type: ndcg_at_5
      value: 32.65
    - type: precision_at_1
      value: 27.454
    - type: precision_at_10
      value: 5.337
    - type: precision_at_100
      value: 0.8250000000000001
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 13.241
    - type: precision_at_5
      value: 8.895999999999999
    - type: recall_at_1
      value: 24.537
    - type: recall_at_10
      value: 44.324999999999996
    - type: recall_at_100
      value: 65.949
    - type: recall_at_1000
      value: 84.017
    - type: recall_at_3
      value: 33.857
    - type: recall_at_5
      value: 37.316
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackTexRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 17.122
    - type: map_at_10
      value: 24.32
    - type: map_at_100
      value: 25.338
    - type: map_at_1000
      value: 25.462
    - type: map_at_3
      value: 22.064
    - type: map_at_5
      value: 23.322000000000003
    - type: mrr_at_1
      value: 20.647
    - type: mrr_at_10
      value: 27.858
    - type: mrr_at_100
      value: 28.743999999999996
    - type: mrr_at_1000
      value: 28.819
    - type: mrr_at_3
      value: 25.769
    - type: mrr_at_5
      value: 26.964
    - type: ndcg_at_1
      value: 20.647
    - type: ndcg_at_10
      value: 28.849999999999998
    - type: ndcg_at_100
      value: 33.849000000000004
    - type: ndcg_at_1000
      value: 36.802
    - type: ndcg_at_3
      value: 24.799
    - type: ndcg_at_5
      value: 26.682
    - type: precision_at_1
      value: 20.647
    - type: precision_at_10
      value: 5.2170000000000005
    - type: precision_at_100
      value: 0.906
    - type: precision_at_1000
      value: 0.134
    - type: precision_at_3
      value: 11.769
    - type: precision_at_5
      value: 8.486
    - type: recall_at_1
      value: 17.122
    - type: recall_at_10
      value: 38.999
    - type: recall_at_100
      value: 61.467000000000006
    - type: recall_at_1000
      value: 82.716
    - type: recall_at_3
      value: 27.601
    - type: recall_at_5
      value: 32.471
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackUnixRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 24.396
    - type: map_at_10
      value: 33.415
    - type: map_at_100
      value: 34.521
    - type: map_at_1000
      value: 34.631
    - type: map_at_3
      value: 30.703999999999997
    - type: map_at_5
      value: 32.166
    - type: mrr_at_1
      value: 28.825
    - type: mrr_at_10
      value: 37.397000000000006
    - type: mrr_at_100
      value: 38.286
    - type: mrr_at_1000
      value: 38.346000000000004
    - type: mrr_at_3
      value: 35.028
    - type: mrr_at_5
      value: 36.32
    - type: ndcg_at_1
      value: 28.825
    - type: ndcg_at_10
      value: 38.656
    - type: ndcg_at_100
      value: 43.856
    - type: ndcg_at_1000
      value: 46.31
    - type: ndcg_at_3
      value: 33.793
    - type: ndcg_at_5
      value: 35.909
    - type: precision_at_1
      value: 28.825
    - type: precision_at_10
      value: 6.567
    - type: precision_at_100
      value: 1.0330000000000001
    - type: precision_at_1000
      value: 0.135
    - type: precision_at_3
      value: 15.516
    - type: precision_at_5
      value: 10.914
    - type: recall_at_1
      value: 24.396
    - type: recall_at_10
      value: 50.747
    - type: recall_at_100
      value: 73.477
    - type: recall_at_1000
      value: 90.801
    - type: recall_at_3
      value: 37.1
    - type: recall_at_5
      value: 42.589
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackWebmastersRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 25.072
    - type: map_at_10
      value: 34.307
    - type: map_at_100
      value: 35.725
    - type: map_at_1000
      value: 35.943999999999996
    - type: map_at_3
      value: 30.906
    - type: map_at_5
      value: 32.818000000000005
    - type: mrr_at_1
      value: 29.644
    - type: mrr_at_10
      value: 38.673
    - type: mrr_at_100
      value: 39.459
    - type: mrr_at_1000
      value: 39.527
    - type: mrr_at_3
      value: 35.771
    - type: mrr_at_5
      value: 37.332
    - type: ndcg_at_1
      value: 29.644
    - type: ndcg_at_10
      value: 40.548
    - type: ndcg_at_100
      value: 45.678999999999995
    - type: ndcg_at_1000
      value: 48.488
    - type: ndcg_at_3
      value: 34.887
    - type: ndcg_at_5
      value: 37.543
    - type: precision_at_1
      value: 29.644
    - type: precision_at_10
      value: 7.688000000000001
    - type: precision_at_100
      value: 1.482
    - type: precision_at_1000
      value: 0.23600000000000002
    - type: precision_at_3
      value: 16.206
    - type: precision_at_5
      value: 12.016
    - type: recall_at_1
      value: 25.072
    - type: recall_at_10
      value: 53.478
    - type: recall_at_100
      value: 76.07300000000001
    - type: recall_at_1000
      value: 93.884
    - type: recall_at_3
      value: 37.583
    - type: recall_at_5
      value: 44.464
  - task:
      type: Retrieval
    dataset:
      type: BeIR/cqadupstack
      name: MTEB CQADupstackWordpressRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 20.712
    - type: map_at_10
      value: 27.467999999999996
    - type: map_at_100
      value: 28.502
    - type: map_at_1000
      value: 28.610000000000003
    - type: map_at_3
      value: 24.887999999999998
    - type: map_at_5
      value: 26.273999999999997
    - type: mrr_at_1
      value: 22.736
    - type: mrr_at_10
      value: 29.553
    - type: mrr_at_100
      value: 30.485
    - type: mrr_at_1000
      value: 30.56
    - type: mrr_at_3
      value: 27.078999999999997
    - type: mrr_at_5
      value: 28.401
    - type: ndcg_at_1
      value: 22.736
    - type: ndcg_at_10
      value: 32.023
    - type: ndcg_at_100
      value: 37.158
    - type: ndcg_at_1000
      value: 39.823
    - type: ndcg_at_3
      value: 26.951999999999998
    - type: ndcg_at_5
      value: 29.281000000000002
    - type: precision_at_1
      value: 22.736
    - type: precision_at_10
      value: 5.213
    - type: precision_at_100
      value: 0.832
    - type: precision_at_1000
      value: 0.116
    - type: precision_at_3
      value: 11.459999999999999
    - type: precision_at_5
      value: 8.244
    - type: recall_at_1
      value: 20.712
    - type: recall_at_10
      value: 44.057
    - type: recall_at_100
      value: 67.944
    - type: recall_at_1000
      value: 87.925
    - type: recall_at_3
      value: 30.305
    - type: recall_at_5
      value: 36.071999999999996
  - task:
      type: Retrieval
    dataset:
      type: climate-fever
      name: MTEB ClimateFEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 10.181999999999999
    - type: map_at_10
      value: 16.66
    - type: map_at_100
      value: 18.273
    - type: map_at_1000
      value: 18.45
    - type: map_at_3
      value: 14.141
    - type: map_at_5
      value: 15.455
    - type: mrr_at_1
      value: 22.15
    - type: mrr_at_10
      value: 32.062000000000005
    - type: mrr_at_100
      value: 33.116
    - type: mrr_at_1000
      value: 33.168
    - type: mrr_at_3
      value: 28.827
    - type: mrr_at_5
      value: 30.892999999999997
    - type: ndcg_at_1
      value: 22.15
    - type: ndcg_at_10
      value: 23.532
    - type: ndcg_at_100
      value: 30.358
    - type: ndcg_at_1000
      value: 33.783
    - type: ndcg_at_3
      value: 19.222
    - type: ndcg_at_5
      value: 20.919999999999998
    - type: precision_at_1
      value: 22.15
    - type: precision_at_10
      value: 7.185999999999999
    - type: precision_at_100
      value: 1.433
    - type: precision_at_1000
      value: 0.207
    - type: precision_at_3
      value: 13.941
    - type: precision_at_5
      value: 10.906
    - type: recall_at_1
      value: 10.181999999999999
    - type: recall_at_10
      value: 28.104000000000003
    - type: recall_at_100
      value: 51.998999999999995
    - type: recall_at_1000
      value: 71.311
    - type: recall_at_3
      value: 17.698
    - type: recall_at_5
      value: 22.262999999999998
  - task:
      type: Retrieval
    dataset:
      type: dbpedia-entity
      name: MTEB DBPedia
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 6.669
    - type: map_at_10
      value: 15.552
    - type: map_at_100
      value: 21.865000000000002
    - type: map_at_1000
      value: 23.268
    - type: map_at_3
      value: 11.309
    - type: map_at_5
      value: 13.084000000000001
    - type: mrr_at_1
      value: 55.50000000000001
    - type: mrr_at_10
      value: 66.46600000000001
    - type: mrr_at_100
      value: 66.944
    - type: mrr_at_1000
      value: 66.956
    - type: mrr_at_3
      value: 64.542
    - type: mrr_at_5
      value: 65.717
    - type: ndcg_at_1
      value: 44.75
    - type: ndcg_at_10
      value: 35.049
    - type: ndcg_at_100
      value: 39.073
    - type: ndcg_at_1000
      value: 46.208
    - type: ndcg_at_3
      value: 39.525
    - type: ndcg_at_5
      value: 37.156
    - type: precision_at_1
      value: 55.50000000000001
    - type: precision_at_10
      value: 27.800000000000004
    - type: precision_at_100
      value: 9.013
    - type: precision_at_1000
      value: 1.8800000000000001
    - type: precision_at_3
      value: 42.667
    - type: precision_at_5
      value: 36.0
    - type: recall_at_1
      value: 6.669
    - type: recall_at_10
      value: 21.811
    - type: recall_at_100
      value: 45.112
    - type: recall_at_1000
      value: 67.806
    - type: recall_at_3
      value: 13.373
    - type: recall_at_5
      value: 16.615
  - task:
      type: Classification
    dataset:
      type: mteb/emotion
      name: MTEB EmotionClassification
      config: default
      split: test
      revision: 4f58c6b202a23cf9a4da393831edf4f9183cad37
    metrics:
    - type: accuracy
      value: 48.769999999999996
    - type: f1
      value: 42.91448356376592
  - task:
      type: Retrieval
    dataset:
      type: fever
      name: MTEB FEVER
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 54.013
    - type: map_at_10
      value: 66.239
    - type: map_at_100
      value: 66.62599999999999
    - type: map_at_1000
      value: 66.644
    - type: map_at_3
      value: 63.965
    - type: map_at_5
      value: 65.45400000000001
    - type: mrr_at_1
      value: 58.221000000000004
    - type: mrr_at_10
      value: 70.43700000000001
    - type: mrr_at_100
      value: 70.744
    - type: mrr_at_1000
      value: 70.75099999999999
    - type: mrr_at_3
      value: 68.284
    - type: mrr_at_5
      value: 69.721
    - type: ndcg_at_1
      value: 58.221000000000004
    - type: ndcg_at_10
      value: 72.327
    - type: ndcg_at_100
      value: 73.953
    - type: ndcg_at_1000
      value: 74.312
    - type: ndcg_at_3
      value: 68.062
    - type: ndcg_at_5
      value: 70.56400000000001
    - type: precision_at_1
      value: 58.221000000000004
    - type: precision_at_10
      value: 9.521
    - type: precision_at_100
      value: 1.045
    - type: precision_at_1000
      value: 0.109
    - type: precision_at_3
      value: 27.348
    - type: precision_at_5
      value: 17.794999999999998
    - type: recall_at_1
      value: 54.013
    - type: recall_at_10
      value: 86.957
    - type: recall_at_100
      value: 93.911
    - type: recall_at_1000
      value: 96.38
    - type: recall_at_3
      value: 75.555
    - type: recall_at_5
      value: 81.671
  - task:
      type: Retrieval
    dataset:
      type: fiqa
      name: MTEB FiQA2018
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 21.254
    - type: map_at_10
      value: 33.723
    - type: map_at_100
      value: 35.574
    - type: map_at_1000
      value: 35.730000000000004
    - type: map_at_3
      value: 29.473
    - type: map_at_5
      value: 31.543
    - type: mrr_at_1
      value: 41.358
    - type: mrr_at_10
      value: 49.498
    - type: mrr_at_100
      value: 50.275999999999996
    - type: mrr_at_1000
      value: 50.308
    - type: mrr_at_3
      value: 47.016000000000005
    - type: mrr_at_5
      value: 48.336
    - type: ndcg_at_1
      value: 41.358
    - type: ndcg_at_10
      value: 41.579
    - type: ndcg_at_100
      value: 48.455
    - type: ndcg_at_1000
      value: 51.165000000000006
    - type: ndcg_at_3
      value: 37.681
    - type: ndcg_at_5
      value: 38.49
    - type: precision_at_1
      value: 41.358
    - type: precision_at_10
      value: 11.543000000000001
    - type: precision_at_100
      value: 1.87
    - type: precision_at_1000
      value: 0.23600000000000002
    - type: precision_at_3
      value: 24.743000000000002
    - type: precision_at_5
      value: 17.994
    - type: recall_at_1
      value: 21.254
    - type: recall_at_10
      value: 48.698
    - type: recall_at_100
      value: 74.588
    - type: recall_at_1000
      value: 91.00200000000001
    - type: recall_at_3
      value: 33.939
    - type: recall_at_5
      value: 39.367000000000004
  - task:
      type: Retrieval
    dataset:
      type: hotpotqa
      name: MTEB HotpotQA
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 35.922
    - type: map_at_10
      value: 52.32599999999999
    - type: map_at_100
      value: 53.18000000000001
    - type: map_at_1000
      value: 53.245
    - type: map_at_3
      value: 49.294
    - type: map_at_5
      value: 51.202999999999996
    - type: mrr_at_1
      value: 71.843
    - type: mrr_at_10
      value: 78.24600000000001
    - type: mrr_at_100
      value: 78.515
    - type: mrr_at_1000
      value: 78.527
    - type: mrr_at_3
      value: 77.17500000000001
    - type: mrr_at_5
      value: 77.852
    - type: ndcg_at_1
      value: 71.843
    - type: ndcg_at_10
      value: 61.379
    - type: ndcg_at_100
      value: 64.535
    - type: ndcg_at_1000
      value: 65.888
    - type: ndcg_at_3
      value: 56.958
    - type: ndcg_at_5
      value: 59.434
    - type: precision_at_1
      value: 71.843
    - type: precision_at_10
      value: 12.686
    - type: precision_at_100
      value: 1.517
    - type: precision_at_1000
      value: 0.16999999999999998
    - type: precision_at_3
      value: 35.778
    - type: precision_at_5
      value: 23.422
    - type: recall_at_1
      value: 35.922
    - type: recall_at_10
      value: 63.43
    - type: recall_at_100
      value: 75.868
    - type: recall_at_1000
      value: 84.88900000000001
    - type: recall_at_3
      value: 53.666000000000004
    - type: recall_at_5
      value: 58.555
  - task:
      type: Classification
    dataset:
      type: mteb/imdb
      name: MTEB ImdbClassification
      config: default
      split: test
      revision: 3d86128a09e091d6018b6d26cad27f2739fc2db7
    metrics:
    - type: accuracy
      value: 79.4408
    - type: ap
      value: 73.52820871620366
    - type: f1
      value: 79.36240238685001
  - task:
      type: Retrieval
    dataset:
      type: msmarco
      name: MTEB MSMARCO
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 21.826999999999998
    - type: map_at_10
      value: 34.04
    - type: map_at_100
      value: 35.226
    - type: map_at_1000
      value: 35.275
    - type: map_at_3
      value: 30.165999999999997
    - type: map_at_5
      value: 32.318000000000005
    - type: mrr_at_1
      value: 22.464000000000002
    - type: mrr_at_10
      value: 34.631
    - type: mrr_at_100
      value: 35.752
    - type: mrr_at_1000
      value: 35.795
    - type: mrr_at_3
      value: 30.798
    - type: mrr_at_5
      value: 32.946999999999996
    - type: ndcg_at_1
      value: 22.464000000000002
    - type: ndcg_at_10
      value: 40.919
    - type: ndcg_at_100
      value: 46.632
    - type: ndcg_at_1000
      value: 47.833
    - type: ndcg_at_3
      value: 32.992
    - type: ndcg_at_5
      value: 36.834
    - type: precision_at_1
      value: 22.464000000000002
    - type: precision_at_10
      value: 6.494
    - type: precision_at_100
      value: 0.9369999999999999
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 14.021
    - type: precision_at_5
      value: 10.347000000000001
    - type: recall_at_1
      value: 21.826999999999998
    - type: recall_at_10
      value: 62.132
    - type: recall_at_100
      value: 88.55199999999999
    - type: recall_at_1000
      value: 97.707
    - type: recall_at_3
      value: 40.541
    - type: recall_at_5
      value: 49.739
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_domain
      name: MTEB MTOPDomainClassification (en)
      config: en
      split: test
      revision: d80d48c1eb48d3562165c59d59d0034df9fff0bf
    metrics:
    - type: accuracy
      value: 95.68399452804377
    - type: f1
      value: 95.25490609832268
  - task:
      type: Classification
    dataset:
      type: mteb/mtop_intent
      name: MTEB MTOPIntentClassification (en)
      config: en
      split: test
      revision: ae001d0e6b1228650b7bd1c2c65fb50ad11a8aba
    metrics:
    - type: accuracy
      value: 83.15321477428182
    - type: f1
      value: 60.35476439087966
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (en)
      config: en
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 71.92669804976462
    - type: f1
      value: 69.22815107207565
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (en)
      config: en
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 74.4855413584398
    - type: f1
      value: 72.92107516103387
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-p2p
      name: MTEB MedrxivClusteringP2P
      config: default
      split: test
      revision: e7a26af6f3ae46b30dde8737f02c07b1505bcc73
    metrics:
    - type: v_measure
      value: 32.412679360205544
  - task:
      type: Clustering
    dataset:
      type: mteb/medrxiv-clustering-s2s
      name: MTEB MedrxivClusteringS2S
      config: default
      split: test
      revision: 35191c8c0dca72d8ff3efcd72aa802307d469663
    metrics:
    - type: v_measure
      value: 28.09211869875204
  - task:
      type: Reranking
    dataset:
      type: mteb/mind_small
      name: MTEB MindSmallReranking
      config: default
      split: test
      revision: 3bdac13927fdc888b903db93b2ffdbd90b295a69
    metrics:
    - type: map
      value: 30.540919056982545
    - type: mrr
      value: 31.529904607063536
  - task:
      type: Retrieval
    dataset:
      type: nfcorpus
      name: MTEB NFCorpus
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 5.745
    - type: map_at_10
      value: 12.013
    - type: map_at_100
      value: 15.040000000000001
    - type: map_at_1000
      value: 16.427
    - type: map_at_3
      value: 8.841000000000001
    - type: map_at_5
      value: 10.289
    - type: mrr_at_1
      value: 45.201
    - type: mrr_at_10
      value: 53.483999999999995
    - type: mrr_at_100
      value: 54.20700000000001
    - type: mrr_at_1000
      value: 54.252
    - type: mrr_at_3
      value: 51.29
    - type: mrr_at_5
      value: 52.73
    - type: ndcg_at_1
      value: 43.808
    - type: ndcg_at_10
      value: 32.445
    - type: ndcg_at_100
      value: 30.031000000000002
    - type: ndcg_at_1000
      value: 39.007
    - type: ndcg_at_3
      value: 37.204
    - type: ndcg_at_5
      value: 35.07
    - type: precision_at_1
      value: 45.201
    - type: precision_at_10
      value: 23.684
    - type: precision_at_100
      value: 7.600999999999999
    - type: precision_at_1000
      value: 2.043
    - type: precision_at_3
      value: 33.953
    - type: precision_at_5
      value: 29.412
    - type: recall_at_1
      value: 5.745
    - type: recall_at_10
      value: 16.168
    - type: recall_at_100
      value: 30.875999999999998
    - type: recall_at_1000
      value: 62.686
    - type: recall_at_3
      value: 9.75
    - type: recall_at_5
      value: 12.413
  - task:
      type: Retrieval
    dataset:
      type: nq
      name: MTEB NQ
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 37.828
    - type: map_at_10
      value: 53.239000000000004
    - type: map_at_100
      value: 54.035999999999994
    - type: map_at_1000
      value: 54.067
    - type: map_at_3
      value: 49.289
    - type: map_at_5
      value: 51.784
    - type: mrr_at_1
      value: 42.497
    - type: mrr_at_10
      value: 55.916999999999994
    - type: mrr_at_100
      value: 56.495
    - type: mrr_at_1000
      value: 56.516999999999996
    - type: mrr_at_3
      value: 52.800000000000004
    - type: mrr_at_5
      value: 54.722
    - type: ndcg_at_1
      value: 42.468
    - type: ndcg_at_10
      value: 60.437
    - type: ndcg_at_100
      value: 63.731
    - type: ndcg_at_1000
      value: 64.41799999999999
    - type: ndcg_at_3
      value: 53.230999999999995
    - type: ndcg_at_5
      value: 57.26
    - type: precision_at_1
      value: 42.468
    - type: precision_at_10 
      value: 9.47
    - type: precision_at_100
      value: 1.1360000000000001
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 23.724999999999998
    - type: precision_at_5
      value: 16.593
    - type: recall_at_1
      value: 37.828
    - type: recall_at_10
      value: 79.538
    - type: recall_at_100
      value: 93.646
    - type: recall_at_1000
      value: 98.72999999999999
    - type: recall_at_3
      value: 61.134
    - type: recall_at_5
      value: 70.377
  - task:
      type: Retrieval
    dataset:
      type: quora
      name: MTEB QuoraRetrieval
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 70.548
    - type: map_at_10
      value: 84.466
    - type: map_at_100
      value: 85.10600000000001
    - type: map_at_1000
      value: 85.123
    - type: map_at_3
      value: 81.57600000000001
    - type: map_at_5
      value: 83.399
    - type: mrr_at_1
      value: 81.24
    - type: mrr_at_10
      value: 87.457
    - type: mrr_at_100
      value: 87.574
    - type: mrr_at_1000
      value: 87.575
    - type: mrr_at_3
      value: 86.507
    - type: mrr_at_5
      value: 87.205
    - type: ndcg_at_1
      value: 81.25
    - type: ndcg_at_10
      value: 88.203
    - type: ndcg_at_100
      value: 89.457
    - type: ndcg_at_1000
      value: 89.563
    - type: ndcg_at_3
      value: 85.465
    - type: ndcg_at_5
      value: 87.007
    - type: precision_at_1
      value: 81.25
    - type: precision_at_10
      value: 13.373
    - type: precision_at_100
      value: 1.5270000000000001
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.417
    - type: precision_at_5
      value: 24.556
    - type: recall_at_1
      value: 70.548
    - type: recall_at_10
      value: 95.208
    - type: recall_at_100
      value: 99.514
    - type: recall_at_1000
      value: 99.988
    - type: recall_at_3
      value: 87.214
    - type: recall_at_5
      value: 91.696
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering
      name: MTEB RedditClustering
      config: default
      split: test
      revision: 24640382cdbf8abc73003fb0fa6d111a705499eb
    metrics:
    - type: v_measure
      value: 53.04822095496839
  - task:
      type: Clustering
    dataset:
      type: mteb/reddit-clustering-p2p
      name: MTEB RedditClusteringP2P
      config: default
      split: test
      revision: 282350215ef01743dc01b456c7f5241fa8937f16
    metrics:
    - type: v_measure
      value: 60.30778476474675
  - task:
      type: Retrieval
    dataset:
      type: scidocs
      name: MTEB SCIDOCS
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 4.692
    - type: map_at_10
      value: 11.766
    - type: map_at_100
      value: 13.904
    - type: map_at_1000
      value: 14.216999999999999
    - type: map_at_3
      value: 8.245
    - type: map_at_5
      value: 9.92
    - type: mrr_at_1
      value: 23.0
    - type: mrr_at_10
      value: 33.78
    - type: mrr_at_100
      value: 34.922
    - type: mrr_at_1000
      value: 34.973
    - type: mrr_at_3
      value: 30.2
    - type: mrr_at_5
      value: 32.565
    - type: ndcg_at_1
      value: 23.0
    - type: ndcg_at_10
      value: 19.863
    - type: ndcg_at_100
      value: 28.141
    - type: ndcg_at_1000
      value: 33.549
    - type: ndcg_at_3
      value: 18.434
    - type: ndcg_at_5
      value: 16.384
    - type: precision_at_1
      value: 23.0
    - type: precision_at_10
      value: 10.39
    - type: precision_at_100
      value: 2.235
    - type: precision_at_1000
      value: 0.35300000000000004
    - type: precision_at_3
      value: 17.133000000000003
    - type: precision_at_5
      value: 14.44
    - type: recall_at_1
      value: 4.692
    - type: recall_at_10
      value: 21.025
    - type: recall_at_100
      value: 45.324999999999996
    - type: recall_at_1000
      value: 71.675
    - type: recall_at_3
      value: 10.440000000000001
    - type: recall_at_5
      value: 14.64
  - task:
      type: STS
    dataset:
      type: mteb/sickr-sts
      name: MTEB SICK-R
      config: default
      split: test
      revision: a6ea5a8cab320b040a23452cc28066d9beae2cee
    metrics:
    - type: cos_sim_pearson
      value: 84.96178184892842
    - type: cos_sim_spearman
      value: 79.6487740813199
    - type: euclidean_pearson
      value: 82.06661161625023
    - type: euclidean_spearman
      value: 79.64876769031183
    - type: manhattan_pearson
      value: 82.07061164575131
    - type: manhattan_spearman
      value: 79.65197039464537
  - task:
      type: STS
    dataset:
      type: mteb/sts12-sts
      name: MTEB STS12
      config: default
      split: test
      revision: a0d554a64d88156834ff5ae9920b964011b16384
    metrics:
    - type: cos_sim_pearson
      value: 84.15305604100027
    - type: cos_sim_spearman
      value: 74.27447427941591
    - type: euclidean_pearson
      value: 80.52737337565307
    - type: euclidean_spearman
      value: 74.27416077132192
    - type: manhattan_pearson
      value: 80.53728571140387
    - type: manhattan_spearman
      value: 74.28853605753457
  - task:
      type: STS
    dataset:
      type: mteb/sts13-sts
      name: MTEB STS13
      config: default
      split: test
      revision: 7e90230a92c190f1bf69ae9002b8cea547a64cca
    metrics:
    - type: cos_sim_pearson
      value: 83.44386080639279
    - type: cos_sim_spearman
      value: 84.17947648159536
    - type: euclidean_pearson
      value: 83.34145388129387
    - type: euclidean_spearman
      value: 84.17947648159536
    - type: manhattan_pearson
      value: 83.30699061927966
    - type: manhattan_spearman
      value: 84.18125737380451
  - task:
      type: STS
    dataset:
      type: mteb/sts14-sts
      name: MTEB STS14
      config: default
      split: test
      revision: 6031580fec1f6af667f0bd2da0a551cf4f0b2375
    metrics:
    - type: cos_sim_pearson
      value: 81.57392220985612
    - type: cos_sim_spearman
      value: 78.80745014464101
    - type: euclidean_pearson
      value: 80.01660371487199
    - type: euclidean_spearman
      value: 78.80741240102256
    - type: manhattan_pearson
      value: 79.96810779507953
    - type: manhattan_spearman
      value: 78.75600400119448
  - task:
      type: STS
    dataset:
      type: mteb/sts15-sts
      name: MTEB STS15
      config: default
      split: test
      revision: ae752c7c21bf194d8b67fd573edf7ae58183cbe3
    metrics:
    - type: cos_sim_pearson
      value: 86.85421063026625
    - type: cos_sim_spearman
      value: 87.55320285299192
    - type: euclidean_pearson
      value: 86.69750143323517
    - type: euclidean_spearman
      value: 87.55320284326378
    - type: manhattan_pearson
      value: 86.63379169960379
    - type: manhattan_spearman
      value: 87.4815029877984
  - task:
      type: STS
    dataset:
      type: mteb/sts16-sts
      name: MTEB STS16
      config: default
      split: test
      revision: 4d8694f8f0e0100860b497b999b3dbed754a0513
    metrics:
    - type: cos_sim_pearson
      value: 84.31314130411842
    - type: cos_sim_spearman
      value: 85.3489588181433
    - type: euclidean_pearson
      value: 84.13240933463535
    - type: euclidean_spearman
      value: 85.34902871403281
    - type: manhattan_pearson
      value: 84.01183086503559
    - type: manhattan_spearman
      value: 85.19316703166102
  - task:
      type: STS
    dataset:
      type: mteb/sts17-crosslingual-sts
      name: MTEB STS17 (en-en)
      config: en-en
      split: test
      revision: af5e6fb845001ecf41f4c1e033ce921939a2a68d
    metrics:
    - type: cos_sim_pearson
      value: 89.09979781689536
    - type: cos_sim_spearman
      value: 88.87813323759015
    - type: euclidean_pearson
      value: 88.65413031123792
    - type: euclidean_spearman
      value: 88.87813323759015
    - type: manhattan_pearson
      value: 88.61818758256024
    - type: manhattan_spearman
      value: 88.81044100494604
  - task:
      type: STS
    dataset:
      type: mteb/sts22-crosslingual-sts
      name: MTEB STS22 (en)
      config: en
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 62.30693258111531
    - type: cos_sim_spearman
      value: 62.195516523251946
    - type: euclidean_pearson
      value: 62.951283701049476
    - type: euclidean_spearman
      value: 62.195516523251946
    - type: manhattan_pearson
      value: 63.068322281439535
    - type: manhattan_spearman
      value: 62.10621171028406
  - task:
      type: STS
    dataset:
      type: mteb/stsbenchmark-sts
      name: MTEB STSBenchmark
      config: default
      split: test
      revision: b0fddb56ed78048fa8b90373c8a3cfc37b684831
    metrics:
    - type: cos_sim_pearson
      value: 84.27092833763909
    - type: cos_sim_spearman
      value: 84.84429717949759
    - type: euclidean_pearson
      value: 84.8516966060792
    - type: euclidean_spearman
      value: 84.84429717949759
    - type: manhattan_pearson
      value: 84.82203139242881
    - type: manhattan_spearman
      value: 84.8358503952945
  - task:
      type: Reranking
    dataset:
      type: mteb/scidocs-reranking
      name: MTEB SciDocsRR
      config: default
      split: test
      revision: d3c5e1fc0b855ab6097bf1cda04dd73947d7caab
    metrics:
    - type: map
      value: 83.10290863981409
    - type: mrr
      value: 95.31168450286097
  - task:
      type: Retrieval
    dataset:
      type: scifact
      name: MTEB SciFact
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 52.161
    - type: map_at_10
      value: 62.138000000000005
    - type: map_at_100
      value: 62.769
    - type: map_at_1000
      value: 62.812
    - type: map_at_3
      value: 59.111000000000004
    - type: map_at_5
      value: 60.995999999999995
    - type: mrr_at_1
      value: 55.333
    - type: mrr_at_10
      value: 63.504000000000005
    - type: mrr_at_100
      value: 64.036
    - type: mrr_at_1000
      value: 64.08
    - type: mrr_at_3
      value: 61.278
    - type: mrr_at_5
      value: 62.778
    - type: ndcg_at_1
      value: 55.333
    - type: ndcg_at_10
      value: 66.678
    - type: ndcg_at_100
      value: 69.415
    - type: ndcg_at_1000
      value: 70.453
    - type: ndcg_at_3
      value: 61.755
    - type: ndcg_at_5
      value: 64.546
    - type: precision_at_1
      value: 55.333
    - type: precision_at_10
      value: 9.033
    - type: precision_at_100
      value: 1.043
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 24.221999999999998
    - type: precision_at_5
      value: 16.333000000000002
    - type: recall_at_1
      value: 52.161
    - type: recall_at_10
      value: 79.156
    - type: recall_at_100
      value: 91.333
    - type: recall_at_1000
      value: 99.333
    - type: recall_at_3
      value: 66.43299999999999
    - type: recall_at_5
      value: 73.272
  - task:
      type: PairClassification
    dataset:
      type: mteb/sprintduplicatequestions-pairclassification
      name: MTEB SprintDuplicateQuestions
      config: default
      split: test
      revision: d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46
    metrics:
    - type: cos_sim_accuracy
      value: 99.81287128712871
    - type: cos_sim_ap
      value: 95.30034785910676
    - type: cos_sim_f1
      value: 90.28629856850716
    - type: cos_sim_precision
      value: 92.36401673640168
    - type: cos_sim_recall
      value: 88.3
    - type: dot_accuracy
      value: 99.81287128712871
    - type: dot_ap
      value: 95.30034785910676
    - type: dot_f1
      value: 90.28629856850716
    - type: dot_precision
      value: 92.36401673640168
    - type: dot_recall
      value: 88.3
    - type: euclidean_accuracy
      value: 99.81287128712871
    - type: euclidean_ap
      value: 95.30034785910676
    - type: euclidean_f1
      value: 90.28629856850716
    - type: euclidean_precision
      value: 92.36401673640168
    - type: euclidean_recall
      value: 88.3
    - type: manhattan_accuracy
      value: 99.80990099009901
    - type: manhattan_ap
      value: 95.26880751950654
    - type: manhattan_f1
      value: 90.22177419354838
    - type: manhattan_precision
      value: 90.95528455284553
    - type: manhattan_recall
      value: 89.5
    - type: max_accuracy
      value: 99.81287128712871
    - type: max_ap
      value: 95.30034785910676
    - type: max_f1
      value: 90.28629856850716
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering
      name: MTEB StackExchangeClustering
      config: default
      split: test
      revision: 6cbc1f7b2bc0622f2e39d2c77fa502909748c259
    metrics:
    - type: v_measure
      value: 58.518662504351184
  - task:
      type: Clustering
    dataset:
      type: mteb/stackexchange-clustering-p2p
      name: MTEB StackExchangeClusteringP2P
      config: default
      split: test
      revision: 815ca46b2622cec33ccafc3735d572c266efdb44
    metrics:
    - type: v_measure
      value: 34.96168178378587
  - task:
      type: Reranking
    dataset:
      type: mteb/stackoverflowdupquestions-reranking
      name: MTEB StackOverflowDupQuestions
      config: default
      split: test
      revision: e185fbe320c72810689fc5848eb6114e1ef5ec69
    metrics:
    - type: map
      value: 52.04862593471896
    - type: mrr
      value: 52.97238402936932
  - task:
      type: Summarization
    dataset:
      type: mteb/summeval
      name: MTEB SummEval
      config: default
      split: test
      revision: cda12ad7615edc362dbf25a00fdd61d3b1eaf93c
    metrics:
    - type: cos_sim_pearson
      value: 30.092545236479946
    - type: cos_sim_spearman
      value: 31.599851000175498
    - type: dot_pearson
      value: 30.092542723901676
    - type: dot_spearman
      value: 31.599851000175498
  - task:
      type: Retrieval
    dataset:
      type: trec-covid
      name: MTEB TRECCOVID
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 0.189
    - type: map_at_10
      value: 1.662
    - type: map_at_100
      value: 9.384
    - type: map_at_1000
      value: 22.669
    - type: map_at_3
      value: 0.5559999999999999
    - type: map_at_5
      value: 0.9039999999999999
    - type: mrr_at_1
      value: 68.0
    - type: mrr_at_10
      value: 81.01899999999999
    - type: mrr_at_100
      value: 81.01899999999999
    - type: mrr_at_1000
      value: 81.01899999999999
    - type: mrr_at_3
      value: 79.333
    - type: mrr_at_5
      value: 80.733
    - type: ndcg_at_1
      value: 63.0
    - type: ndcg_at_10
      value: 65.913
    - type: ndcg_at_100
      value: 51.895
    - type: ndcg_at_1000
      value: 46.967
    - type: ndcg_at_3
      value: 65.49199999999999
    - type: ndcg_at_5
      value: 66.69699999999999
    - type: precision_at_1
      value: 68.0
    - type: precision_at_10
      value: 71.6
    - type: precision_at_100
      value: 53.66
    - type: precision_at_1000
      value: 21.124000000000002
    - type: precision_at_3
      value: 72.667
    - type: precision_at_5
      value: 74.0
    - type: recall_at_1
      value: 0.189
    - type: recall_at_10
      value: 1.913
    - type: recall_at_100
      value: 12.601999999999999
    - type: recall_at_1000
      value: 44.296
    - type: recall_at_3
      value: 0.605
    - type: recall_at_5
      value: 1.018
  - task:
      type: Retrieval
    dataset:
      type: webis-touche2020
      name: MTEB Touche2020
      config: default
      split: test
      revision: None
    metrics:
    - type: map_at_1
      value: 2.701
    - type: map_at_10
      value: 10.445
    - type: map_at_100
      value: 17.324
    - type: map_at_1000
      value: 19.161
    - type: map_at_3
      value: 5.497
    - type: map_at_5
      value: 7.278
    - type: mrr_at_1
      value: 30.612000000000002
    - type: mrr_at_10
      value: 45.534
    - type: mrr_at_100
      value: 45.792
    - type: mrr_at_1000
      value: 45.806999999999995
    - type: mrr_at_3
      value: 37.755
    - type: mrr_at_5
      value: 43.469
    - type: ndcg_at_1
      value: 26.531
    - type: ndcg_at_10
      value: 26.235000000000003
    - type: ndcg_at_100
      value: 39.17
    - type: ndcg_at_1000
      value: 51.038
    - type: ndcg_at_3
      value: 23.625
    - type: ndcg_at_5
      value: 24.338
    - type: precision_at_1
      value: 30.612000000000002
    - type: precision_at_10
      value: 24.285999999999998
    - type: precision_at_100
      value: 8.224
    - type: precision_at_1000
      value: 1.6179999999999999
    - type: precision_at_3
      value: 24.490000000000002
    - type: precision_at_5
      value: 24.898
    - type: recall_at_1
      value: 2.701
    - type: recall_at_10
      value: 17.997
    - type: recall_at_100
      value: 51.766999999999996
    - type: recall_at_1000
      value: 87.863
    - type: recall_at_3
      value: 6.295000000000001
    - type: recall_at_5
      value: 9.993
  - task:
      type: Classification
    dataset:
      type: mteb/toxic_conversations_50k
      name: MTEB ToxicConversationsClassification
      config: default
      split: test
      revision: d7c0de2777da35d6aae2200a62c6e0e5af397c4c
    metrics:
    - type: accuracy
      value: 73.3474
    - type: ap
      value: 15.393431414459924
    - type: f1
      value: 56.466681887882416
  - task:
      type: Classification
    dataset:
      type: mteb/tweet_sentiment_extraction
      name: MTEB TweetSentimentExtractionClassification
      config: default
      split: test
      revision: d604517c81ca91fe16a244d1248fc021f9ecee7a
    metrics:
    - type: accuracy
      value: 62.062818336163
    - type: f1
      value: 62.11230840463252
  - task:
      type: Clustering
    dataset:
      type: mteb/twentynewsgroups-clustering
      name: MTEB TwentyNewsgroupsClustering
      config: default
      split: test
      revision: 6125ec4e24fa026cec8a478383ee943acfbd5449
    metrics:
    - type: v_measure
      value: 42.464892820845115
  - task:
      type: PairClassification
    dataset:
      type: mteb/twittersemeval2015-pairclassification
      name: MTEB TwitterSemEval2015
      config: default
      split: test
      revision: 70970daeab8776df92f5ea462b6173c0b46fd2d1
    metrics:
    - type: cos_sim_accuracy
      value: 86.15962329379508
    - type: cos_sim_ap
      value: 74.73674057919256
    - type: cos_sim_f1
      value: 68.81245642574947
    - type: cos_sim_precision
      value: 61.48255813953488
    - type: cos_sim_recall
      value: 78.12664907651715
    - type: dot_accuracy
      value: 86.15962329379508
    - type: dot_ap
      value: 74.7367634988281
    - type: dot_f1
      value: 68.81245642574947
    - type: dot_precision
      value: 61.48255813953488
    - type: dot_recall
      value: 78.12664907651715
    - type: euclidean_accuracy
      value: 86.15962329379508
    - type: euclidean_ap
      value: 74.7367761466634
    - type: euclidean_f1
      value: 68.81245642574947
    - type: euclidean_precision
      value: 61.48255813953488
    - type: euclidean_recall
      value: 78.12664907651715
    - type: manhattan_accuracy
      value: 86.21326816474935
    - type: manhattan_ap
      value: 74.64416473733951
    - type: manhattan_f1
      value: 68.80924855491331
    - type: manhattan_precision
      value: 61.23456790123457
    - type: manhattan_recall
      value: 78.52242744063325
    - type: max_accuracy
      value: 86.21326816474935
    - type: max_ap
      value: 74.7367761466634
    - type: max_f1
      value: 68.81245642574947
  - task:
      type: PairClassification
    dataset:
      type: mteb/twitterurlcorpus-pairclassification
      name: MTEB TwitterURLCorpus
      config: default
      split: test
      revision: 8b6510b0b1fa4e4c4f879467980e9be563ec1cdf
    metrics:
    - type: cos_sim_accuracy
      value: 88.97620988085536
    - type: cos_sim_ap
      value: 86.08680845745758
    - type: cos_sim_f1
      value: 78.02793637114438
    - type: cos_sim_precision
      value: 73.11082699683736
    - type: cos_sim_recall
      value: 83.65414228518632
    - type: dot_accuracy
      value: 88.97620988085536
    - type: dot_ap
      value: 86.08681149437946
    - type: dot_f1
      value: 78.02793637114438
    - type: dot_precision
      value: 73.11082699683736
    - type: dot_recall
      value: 83.65414228518632
    - type: euclidean_accuracy
      value: 88.97620988085536
    - type: euclidean_ap
      value: 86.08681215460771
    - type: euclidean_f1
      value: 78.02793637114438
    - type: euclidean_precision
      value: 73.11082699683736
    - type: euclidean_recall
      value: 83.65414228518632
    - type: manhattan_accuracy
      value: 88.88888888888889
    - type: manhattan_ap
      value: 86.02916327562438
    - type: manhattan_f1
      value: 78.02063045516843
    - type: manhattan_precision
      value: 73.38851947346994
    - type: manhattan_recall
      value: 83.2768709578072
    - type: max_accuracy
      value: 88.97620988085536
    - type: max_ap
      value: 86.08681215460771
    - type: max_f1
      value: 78.02793637114438
---
<!-- TODO: add evaluation results here -->
<br><br>

<p align="center">
<img src="https://github.com/jina-ai/finetuner/blob/main/docs/_static/finetuner-logo-ani.svg?raw=true" alt="Finetuner logo: Finetuner helps you to create experiments in order to improve embeddings on search tasks. It accompanies you to deliver the last mile of performance-tuning for neural search applications." width="150px">
</p>


<p align="center">
<b>The text embedding set trained by <a href="https://jina.ai/"><b>Jina AI</b></a>, <a href="https://github.com/jina-ai/finetuner"><b>Finetuner</b></a> team.</b>
</p>


## Intended Usage & Model Info

`jina-embeddings-v2-base-en` is an English, monolingual **embedding model** supporting **8192 sequence length**.
It is based on a Bert architecture (JinaBert) that supports the symmetric bidirectional variant of [ALiBi](https://arxiv.org/abs/2108.12409) to allow longer sequence length.
The backbone `jina-bert-v2-base-en` is pretrained on the C4 dataset.
The model is further trained on Jina AI's collection of more than 400 millions of sentence pairs and hard negatives.
These pairs were obtained from various domains and were carefully selected through a thorough cleaning process.

The embedding model was trained using 512 sequence length, but extrapolates to 8k sequence length (or even longer) thanks to ALiBi.
This makes our model useful for a range of use cases, especially when processing long documents is needed, including long document retrieval, semantic textual similarity, text reranking, recommendation, RAG and LLM-based generative search, etc.

With a standard size of 137 million parameters, the model enables fast inference while delivering better performance than our small model. It is recommended to use a single GPU for inference.
Additionally, we provide the following embedding models:

**V1 (Based on T5, 512 Seq)**

- [`jina-embeddings-v1-small-en`](https://huggingface.co/jinaai/jina-embedding-s-en-v1): 35 million parameters.
- [`jina-embeddings-v1-base-en`](https://huggingface.co/jinaai/jina-embedding-b-en-v1): 110 million parameters.
- [`jina-embeddings-v1-large-en`](https://huggingface.co/jinaai/jina-embedding-l-en-v1): 330 million parameters.

**V2 (Based on JinaBert, 8k Seq)**

- [`jina-embeddings-v2-small-en`](https://huggingface.co/jinaai/jina-embeddings-v2-small-en): 33 million parameters.
- [`jina-embeddings-v2-base-en`](https://huggingface.co/jinaai/jina-embeddings-v2-base-en): 137 million parameters **(you are here)**.
- [`jina-embeddings-v2-large-en`](): 435 million parameters (releasing soon).

## Data & Parameters

Jina Embeddings V2 [technical report](https://arxiv.org/abs/2310.19923)

## Usage

You can use Jina Embedding models directly from transformers package:
```python
!pip install transformers
from transformers import AutoModel
from numpy.linalg import norm

cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
embeddings = model.encode(['How is the weather today?', 'What is the current weather like today?'])
print(cos_sim(embeddings[0], embeddings[1]))
```

If you only want to handle shorter sequence, such as 2k, pass the `max_length` parameter to the `encode` function:

```python
embeddings = model.encode(
    ['Very long ... document'],
    max_length=2048
)
```

## Fully-managed Embeddings Service

Alternatively, you can use Jina AI's [Embedding platform](https://jina.ai/embeddings/) for fully-managed access to Jina Embeddings models.

## Use Jina Embeddings for RAG

Jina Embeddings are very effective for retrieval augmented generation (RAG).
Ravi Theja wrote a [blog post](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83) on using Jina Embeddings together with [LLama Index](https://github.com/run-llama/llama_index) for RAG:

<img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*ZP2RVejCZovF3FDCg-Bx3A.png" width="780px">


## Plans

The development of new bilingual models is currently underway. We will be targeting mainly the German and Spanish languages.
The upcoming models will be called `jina-embeddings-v2-base-de/es`.

## Contact

Join our [Discord community](https://discord.jina.ai) and chat with other community members about ideas.

## Citation

If you find Jina Embeddings useful in your research, please cite the following paper:

```
@misc{günther2023jina,
      title={Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents}, 
      author={Michael Günther and Jackmin Ong and Isabelle Mohr and Alaeddine Abdessalem and Tanguy Abel and Mohammad Kalim Akram and Susana Guzman and Georgios Mastrapas and Saba Sturua and Bo Wang and Maximilian Werk and Nan Wang and Han Xiao},
      year={2023},
      eprint={2310.19923},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

<!--
``` latex
@misc{günther2023jina,
      title={Beyond the 512-Token Barrier: Training General-Purpose Text
Embeddings for Large Documents}, 
      author={Michael Günther and Jackmin Ong and Isabelle Mohr and Alaeddine Abdessalem and Tanguy Abel and Mohammad Kalim Akram and Susana Guzman and Georgios Mastrapas and Saba Sturua and Bo Wang},
      year={2023},
      eprint={2307.11224},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{günther2023jina,
      title={Jina Embeddings: A Novel Set of High-Performance Sentence Embedding Models}, 
      author={Michael Günther and Louis Milliken and Jonathan Geuter and Georgios Mastrapas and Bo Wang and Han Xiao},
      year={2023},
      eprint={2307.11224},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
-->