---
tags:
  - finetuner
  - mteb
  - sentence-transformers
  - feature-extraction
  - sentence-similarity
datasets:
  - jinaai/negation-dataset
language: en
inference: false
license: apache-2.0
model-index:
- name: jina-embedding-s-en-v2
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
      value: 71.35820895522387
    - type: ap
      value: 33.99931933598115
    - type: f1
      value: 65.3853685535555
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
      value: 82.90140000000001
    - type: ap
      value: 78.01434597815617
    - type: f1
      value: 82.83357802722676
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
      value: 40.88999999999999
    - type: f1
      value: 39.209432767163456
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
      value: 23.257
    - type: map_at_10
      value: 37.946000000000005
    - type: map_at_100
      value: 39.17
    - type: map_at_1000
      value: 39.181
    - type: map_at_3
      value: 32.99
    - type: map_at_5
      value: 35.467999999999996
    - type: mrr_at_1
      value: 23.541999999999998
    - type: mrr_at_10
      value: 38.057
    - type: mrr_at_100
      value: 39.289
    - type: mrr_at_1000
      value: 39.299
    - type: mrr_at_3
      value: 33.096
    - type: mrr_at_5
      value: 35.628
    - type: ndcg_at_1
      value: 23.257
    - type: ndcg_at_10
      value: 46.729
    - type: ndcg_at_100
      value: 51.900999999999996
    - type: ndcg_at_1000
      value: 52.16
    - type: ndcg_at_3
      value: 36.323
    - type: ndcg_at_5
      value: 40.766999999999996
    - type: precision_at_1
      value: 23.257
    - type: precision_at_10
      value: 7.510999999999999
    - type: precision_at_100
      value: 0.976
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 15.339
    - type: precision_at_5
      value: 11.350999999999999
    - type: recall_at_1
      value: 23.257
    - type: recall_at_10
      value: 75.107
    - type: recall_at_100
      value: 97.58200000000001
    - type: recall_at_1000
      value: 99.57300000000001
    - type: recall_at_3
      value: 46.017
    - type: recall_at_5
      value: 56.757000000000005
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
      value: 44.02420878391967
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
      value: 35.16136856000258
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
      value: 59.61809790513646
    - type: mrr
      value: 73.07215406938397
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
      value: 82.0167350090749
    - type: cos_sim_spearman
      value: 80.51569002630401
    - type: euclidean_pearson
      value: 81.46820525099726
    - type: euclidean_spearman
      value: 80.51569002630401
    - type: manhattan_pearson
      value: 81.35596555056757
    - type: manhattan_spearman
      value: 80.12592210903303
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
      value: 78.25
    - type: f1
      value: 77.34950913540605
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
      value: 35.57238596005698
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
      value: 29.066444306196683
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
      value: 31.891000000000002
    - type: map_at_10
      value: 42.772
    - type: map_at_100
      value: 44.108999999999995
    - type: map_at_1000
      value: 44.236
    - type: map_at_3
      value: 39.289
    - type: map_at_5
      value: 41.113
    - type: mrr_at_1
      value: 39.342
    - type: mrr_at_10
      value: 48.852000000000004
    - type: mrr_at_100
      value: 49.534
    - type: mrr_at_1000
      value: 49.582
    - type: mrr_at_3
      value: 46.089999999999996
    - type: mrr_at_5
      value: 47.685
    - type: ndcg_at_1
      value: 39.342
    - type: ndcg_at_10
      value: 48.988
    - type: ndcg_at_100
      value: 53.854
    - type: ndcg_at_1000
      value: 55.955
    - type: ndcg_at_3
      value: 43.877
    - type: ndcg_at_5
      value: 46.027
    - type: precision_at_1
      value: 39.342
    - type: precision_at_10
      value: 9.285
    - type: precision_at_100
      value: 1.488
    - type: precision_at_1000
      value: 0.194
    - type: precision_at_3
      value: 20.696
    - type: precision_at_5
      value: 14.878
    - type: recall_at_1
      value: 31.891000000000002
    - type: recall_at_10
      value: 60.608
    - type: recall_at_100
      value: 81.025
    - type: recall_at_1000
      value: 94.883
    - type: recall_at_3
      value: 45.694
    - type: recall_at_5
      value: 51.684
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
      value: 28.778
    - type: map_at_10
      value: 37.632
    - type: map_at_100
      value: 38.800000000000004
    - type: map_at_1000
      value: 38.934999999999995
    - type: map_at_3
      value: 35.293
    - type: map_at_5
      value: 36.547000000000004
    - type: mrr_at_1
      value: 35.35
    - type: mrr_at_10
      value: 42.936
    - type: mrr_at_100
      value: 43.69
    - type: mrr_at_1000
      value: 43.739
    - type: mrr_at_3
      value: 41.062
    - type: mrr_at_5
      value: 42.097
    - type: ndcg_at_1
      value: 35.35
    - type: ndcg_at_10
      value: 42.528
    - type: ndcg_at_100
      value: 46.983000000000004
    - type: ndcg_at_1000
      value: 49.187999999999995
    - type: ndcg_at_3
      value: 39.271
    - type: ndcg_at_5
      value: 40.654
    - type: precision_at_1
      value: 35.35
    - type: precision_at_10
      value: 7.828
    - type: precision_at_100
      value: 1.3010000000000002
    - type: precision_at_1000
      value: 0.17700000000000002
    - type: precision_at_3
      value: 18.96
    - type: precision_at_5
      value: 13.120999999999999
    - type: recall_at_1
      value: 28.778
    - type: recall_at_10
      value: 50.775000000000006
    - type: recall_at_100
      value: 69.66799999999999
    - type: recall_at_1000
      value: 83.638
    - type: recall_at_3
      value: 40.757
    - type: recall_at_5
      value: 44.86
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
      value: 37.584
    - type: map_at_10
      value: 49.69
    - type: map_at_100
      value: 50.639
    - type: map_at_1000
      value: 50.702999999999996
    - type: map_at_3
      value: 46.61
    - type: map_at_5
      value: 48.486000000000004
    - type: mrr_at_1
      value: 43.009
    - type: mrr_at_10
      value: 52.949999999999996
    - type: mrr_at_100
      value: 53.618
    - type: mrr_at_1000
      value: 53.65299999999999
    - type: mrr_at_3
      value: 50.605999999999995
    - type: mrr_at_5
      value: 52.095
    - type: ndcg_at_1
      value: 43.009
    - type: ndcg_at_10
      value: 55.278000000000006
    - type: ndcg_at_100
      value: 59.134
    - type: ndcg_at_1000
      value: 60.528999999999996
    - type: ndcg_at_3
      value: 50.184
    - type: ndcg_at_5
      value: 52.919000000000004
    - type: precision_at_1
      value: 43.009
    - type: precision_at_10
      value: 8.821
    - type: precision_at_100
      value: 1.161
    - type: precision_at_1000
      value: 0.133
    - type: precision_at_3
      value: 22.424
    - type: precision_at_5
      value: 15.436
    - type: recall_at_1
      value: 37.584
    - type: recall_at_10
      value: 68.514
    - type: recall_at_100
      value: 85.099
    - type: recall_at_1000
      value: 95.123
    - type: recall_at_3
      value: 55.007
    - type: recall_at_5
      value: 61.714999999999996
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
      value: 24.7
    - type: map_at_10
      value: 32.804
    - type: map_at_100
      value: 33.738
    - type: map_at_1000
      value: 33.825
    - type: map_at_3
      value: 30.639
    - type: map_at_5
      value: 31.781
    - type: mrr_at_1
      value: 26.328000000000003
    - type: mrr_at_10
      value: 34.679
    - type: mrr_at_100
      value: 35.510000000000005
    - type: mrr_at_1000
      value: 35.577999999999996
    - type: mrr_at_3
      value: 32.58
    - type: mrr_at_5
      value: 33.687
    - type: ndcg_at_1
      value: 26.328000000000003
    - type: ndcg_at_10
      value: 37.313
    - type: ndcg_at_100
      value: 42.004000000000005
    - type: ndcg_at_1000
      value: 44.232
    - type: ndcg_at_3
      value: 33.076
    - type: ndcg_at_5
      value: 34.966
    - type: precision_at_1
      value: 26.328000000000003
    - type: precision_at_10
      value: 5.627
    - type: precision_at_100
      value: 0.8410000000000001
    - type: precision_at_1000
      value: 0.106
    - type: precision_at_3
      value: 14.011000000000001
    - type: precision_at_5
      value: 9.582
    - type: recall_at_1
      value: 24.7
    - type: recall_at_10
      value: 49.324
    - type: recall_at_100
      value: 71.018
    - type: recall_at_1000
      value: 87.905
    - type: recall_at_3
      value: 37.7
    - type: recall_at_5
      value: 42.281
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
      value: 14.350999999999999
    - type: map_at_10
      value: 21.745
    - type: map_at_100
      value: 22.731
    - type: map_at_1000
      value: 22.852
    - type: map_at_3
      value: 19.245
    - type: map_at_5
      value: 20.788
    - type: mrr_at_1
      value: 18.159
    - type: mrr_at_10
      value: 25.833000000000002
    - type: mrr_at_100
      value: 26.728
    - type: mrr_at_1000
      value: 26.802
    - type: mrr_at_3
      value: 23.383000000000003
    - type: mrr_at_5
      value: 24.887999999999998
    - type: ndcg_at_1
      value: 18.159
    - type: ndcg_at_10
      value: 26.518000000000004
    - type: ndcg_at_100
      value: 31.473000000000003
    - type: ndcg_at_1000
      value: 34.576
    - type: ndcg_at_3
      value: 21.907
    - type: ndcg_at_5
      value: 24.39
    - type: precision_at_1
      value: 18.159
    - type: precision_at_10
      value: 4.938
    - type: precision_at_100
      value: 0.853
    - type: precision_at_1000
      value: 0.125
    - type: precision_at_3
      value: 10.655000000000001
    - type: precision_at_5
      value: 7.985
    - type: recall_at_1
      value: 14.350999999999999
    - type: recall_at_10
      value: 37.284
    - type: recall_at_100
      value: 59.11300000000001
    - type: recall_at_1000
      value: 81.634
    - type: recall_at_3
      value: 24.753
    - type: recall_at_5
      value: 30.979
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
      value: 26.978
    - type: map_at_10
      value: 36.276
    - type: map_at_100
      value: 37.547000000000004
    - type: map_at_1000
      value: 37.678
    - type: map_at_3
      value: 33.674
    - type: map_at_5
      value: 35.119
    - type: mrr_at_1
      value: 32.916000000000004
    - type: mrr_at_10
      value: 41.798
    - type: mrr_at_100
      value: 42.72
    - type: mrr_at_1000
      value: 42.778
    - type: mrr_at_3
      value: 39.493
    - type: mrr_at_5
      value: 40.927
    - type: ndcg_at_1
      value: 32.916000000000004
    - type: ndcg_at_10
      value: 41.81
    - type: ndcg_at_100
      value: 47.284
    - type: ndcg_at_1000
      value: 49.702
    - type: ndcg_at_3
      value: 37.486999999999995
    - type: ndcg_at_5
      value: 39.597
    - type: precision_at_1
      value: 32.916000000000004
    - type: precision_at_10
      value: 7.411
    - type: precision_at_100
      value: 1.189
    - type: precision_at_1000
      value: 0.158
    - type: precision_at_3
      value: 17.581
    - type: precision_at_5
      value: 12.397
    - type: recall_at_1
      value: 26.978
    - type: recall_at_10
      value: 52.869
    - type: recall_at_100
      value: 75.78399999999999
    - type: recall_at_1000
      value: 91.545
    - type: recall_at_3
      value: 40.717
    - type: recall_at_5
      value: 46.168
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
      value: 24.641
    - type: map_at_10
      value: 32.916000000000004
    - type: map_at_100
      value: 34.165
    - type: map_at_1000
      value: 34.286
    - type: map_at_3
      value: 30.335
    - type: map_at_5
      value: 31.569000000000003
    - type: mrr_at_1
      value: 30.593999999999998
    - type: mrr_at_10
      value: 38.448
    - type: mrr_at_100
      value: 39.299
    - type: mrr_at_1000
      value: 39.362
    - type: mrr_at_3
      value: 36.244
    - type: mrr_at_5
      value: 37.232
    - type: ndcg_at_1
      value: 30.593999999999998
    - type: ndcg_at_10
      value: 38.2
    - type: ndcg_at_100
      value: 43.742
    - type: ndcg_at_1000
      value: 46.217000000000006
    - type: ndcg_at_3
      value: 33.925
    - type: ndcg_at_5
      value: 35.394
    - type: precision_at_1
      value: 30.593999999999998
    - type: precision_at_10
      value: 6.895
    - type: precision_at_100
      value: 1.1320000000000001
    - type: precision_at_1000
      value: 0.153
    - type: precision_at_3
      value: 16.096
    - type: precision_at_5
      value: 11.05
    - type: recall_at_1
      value: 24.641
    - type: recall_at_10
      value: 48.588
    - type: recall_at_100
      value: 72.841
    - type: recall_at_1000
      value: 89.535
    - type: recall_at_3
      value: 36.087
    - type: recall_at_5
      value: 40.346
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
      value: 24.79425
    - type: map_at_10
      value: 33.12033333333333
    - type: map_at_100
      value: 34.221333333333334
    - type: map_at_1000
      value: 34.3435
    - type: map_at_3
      value: 30.636583333333338
    - type: map_at_5
      value: 31.974083333333326
    - type: mrr_at_1
      value: 29.242416666666664
    - type: mrr_at_10
      value: 37.11675
    - type: mrr_at_100
      value: 37.93783333333334
    - type: mrr_at_1000
      value: 38.003083333333336
    - type: mrr_at_3
      value: 34.904666666666664
    - type: mrr_at_5
      value: 36.12916666666667
    - type: ndcg_at_1
      value: 29.242416666666664
    - type: ndcg_at_10
      value: 38.03416666666667
    - type: ndcg_at_100
      value: 42.86674999999999
    - type: ndcg_at_1000
      value: 45.34550000000001
    - type: ndcg_at_3
      value: 33.76466666666666
    - type: ndcg_at_5
      value: 35.668666666666674
    - type: precision_at_1
      value: 29.242416666666664
    - type: precision_at_10
      value: 6.589833333333334
    - type: precision_at_100
      value: 1.0693333333333332
    - type: precision_at_1000
      value: 0.14641666666666667
    - type: precision_at_3
      value: 15.430749999999998
    - type: precision_at_5
      value: 10.833833333333333
    - type: recall_at_1
      value: 24.79425
    - type: recall_at_10
      value: 48.582916666666655
    - type: recall_at_100
      value: 69.88499999999999
    - type: recall_at_1000
      value: 87.211
    - type: recall_at_3
      value: 36.625499999999995
    - type: recall_at_5
      value: 41.553999999999995
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
      value: 22.767
    - type: map_at_10
      value: 28.450999999999997
    - type: map_at_100
      value: 29.332
    - type: map_at_1000
      value: 29.426000000000002
    - type: map_at_3
      value: 26.379
    - type: map_at_5
      value: 27.584999999999997
    - type: mrr_at_1
      value: 25.46
    - type: mrr_at_10
      value: 30.974
    - type: mrr_at_100
      value: 31.784000000000002
    - type: mrr_at_1000
      value: 31.857999999999997
    - type: mrr_at_3
      value: 28.962
    - type: mrr_at_5
      value: 30.066
    - type: ndcg_at_1
      value: 25.46
    - type: ndcg_at_10
      value: 32.041
    - type: ndcg_at_100
      value: 36.522
    - type: ndcg_at_1000
      value: 39.101
    - type: ndcg_at_3
      value: 28.152
    - type: ndcg_at_5
      value: 30.03
    - type: precision_at_1
      value: 25.46
    - type: precision_at_10
      value: 4.893
    - type: precision_at_100
      value: 0.77
    - type: precision_at_1000
      value: 0.107
    - type: precision_at_3
      value: 11.605
    - type: precision_at_5
      value: 8.19
    - type: recall_at_1
      value: 22.767
    - type: recall_at_10
      value: 40.71
    - type: recall_at_100
      value: 61.334999999999994
    - type: recall_at_1000
      value: 80.567
    - type: recall_at_3
      value: 30.198000000000004
    - type: recall_at_5
      value: 34.803
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
      value: 16.722
    - type: map_at_10
      value: 22.794
    - type: map_at_100
      value: 23.7
    - type: map_at_1000
      value: 23.822
    - type: map_at_3
      value: 20.781
    - type: map_at_5
      value: 22.024
    - type: mrr_at_1
      value: 20.061999999999998
    - type: mrr_at_10
      value: 26.346999999999998
    - type: mrr_at_100
      value: 27.153
    - type: mrr_at_1000
      value: 27.233
    - type: mrr_at_3
      value: 24.375
    - type: mrr_at_5
      value: 25.593
    - type: ndcg_at_1
      value: 20.061999999999998
    - type: ndcg_at_10
      value: 26.785999999999998
    - type: ndcg_at_100
      value: 31.319999999999997
    - type: ndcg_at_1000
      value: 34.346
    - type: ndcg_at_3
      value: 23.219
    - type: ndcg_at_5
      value: 25.107000000000003
    - type: precision_at_1
      value: 20.061999999999998
    - type: precision_at_10
      value: 4.78
    - type: precision_at_100
      value: 0.83
    - type: precision_at_1000
      value: 0.125
    - type: precision_at_3
      value: 10.874
    - type: precision_at_5
      value: 7.956
    - type: recall_at_1
      value: 16.722
    - type: recall_at_10
      value: 35.204
    - type: recall_at_100
      value: 55.797
    - type: recall_at_1000
      value: 77.689
    - type: recall_at_3
      value: 25.245
    - type: recall_at_5
      value: 30.115
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
      value: 24.842
    - type: map_at_10
      value: 32.917
    - type: map_at_100
      value: 33.961000000000006
    - type: map_at_1000
      value: 34.069
    - type: map_at_3
      value: 30.595
    - type: map_at_5
      value: 31.837
    - type: mrr_at_1
      value: 29.011
    - type: mrr_at_10
      value: 36.977
    - type: mrr_at_100
      value: 37.814
    - type: mrr_at_1000
      value: 37.885999999999996
    - type: mrr_at_3
      value: 34.966
    - type: mrr_at_5
      value: 36.043
    - type: ndcg_at_1
      value: 29.011
    - type: ndcg_at_10
      value: 37.735
    - type: ndcg_at_100
      value: 42.683
    - type: ndcg_at_1000
      value: 45.198
    - type: ndcg_at_3
      value: 33.650000000000006
    - type: ndcg_at_5
      value: 35.386
    - type: precision_at_1
      value: 29.011
    - type: precision_at_10
      value: 6.259
    - type: precision_at_100
      value: 0.984
    - type: precision_at_1000
      value: 0.13
    - type: precision_at_3
      value: 15.329999999999998
    - type: precision_at_5
      value: 10.541
    - type: recall_at_1
      value: 24.842
    - type: recall_at_10
      value: 48.304
    - type: recall_at_100
      value: 70.04899999999999
    - type: recall_at_1000
      value: 87.82600000000001
    - type: recall_at_3
      value: 36.922
    - type: recall_at_5
      value: 41.449999999999996
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
      value: 24.252000000000002
    - type: map_at_10
      value: 32.293
    - type: map_at_100
      value: 33.816
    - type: map_at_1000
      value: 34.053
    - type: map_at_3
      value: 29.781999999999996
    - type: map_at_5
      value: 31.008000000000003
    - type: mrr_at_1
      value: 29.051
    - type: mrr_at_10
      value: 36.722
    - type: mrr_at_100
      value: 37.663000000000004
    - type: mrr_at_1000
      value: 37.734
    - type: mrr_at_3
      value: 34.354
    - type: mrr_at_5
      value: 35.609
    - type: ndcg_at_1
      value: 29.051
    - type: ndcg_at_10
      value: 37.775999999999996
    - type: ndcg_at_100
      value: 43.221
    - type: ndcg_at_1000
      value: 46.116
    - type: ndcg_at_3
      value: 33.403
    - type: ndcg_at_5
      value: 35.118
    - type: precision_at_1
      value: 29.051
    - type: precision_at_10
      value: 7.332
    - type: precision_at_100
      value: 1.49
    - type: precision_at_1000
      value: 0.23600000000000002
    - type: precision_at_3
      value: 15.415000000000001
    - type: precision_at_5
      value: 11.107
    - type: recall_at_1
      value: 24.252000000000002
    - type: recall_at_10
      value: 47.861
    - type: recall_at_100
      value: 72.21600000000001
    - type: recall_at_1000
      value: 90.886
    - type: recall_at_3
      value: 35.533
    - type: recall_at_5
      value: 39.959
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
      value: 20.025000000000002
    - type: map_at_10
      value: 27.154
    - type: map_at_100
      value: 28.118
    - type: map_at_1000
      value: 28.237000000000002
    - type: map_at_3
      value: 25.017
    - type: map_at_5
      value: 25.832
    - type: mrr_at_1
      value: 21.627
    - type: mrr_at_10
      value: 28.884999999999998
    - type: mrr_at_100
      value: 29.741
    - type: mrr_at_1000
      value: 29.831999999999997
    - type: mrr_at_3
      value: 26.741
    - type: mrr_at_5
      value: 27.628000000000004
    - type: ndcg_at_1
      value: 21.627
    - type: ndcg_at_10
      value: 31.436999999999998
    - type: ndcg_at_100
      value: 36.181000000000004
    - type: ndcg_at_1000
      value: 38.986
    - type: ndcg_at_3
      value: 27.025
    - type: ndcg_at_5
      value: 28.436
    - type: precision_at_1
      value: 21.627
    - type: precision_at_10
      value: 5.009
    - type: precision_at_100
      value: 0.7929999999999999
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 11.522
    - type: precision_at_5
      value: 7.763000000000001
    - type: recall_at_1
      value: 20.025000000000002
    - type: recall_at_10
      value: 42.954
    - type: recall_at_100
      value: 64.67500000000001
    - type: recall_at_1000
      value: 85.301
    - type: recall_at_3
      value: 30.892999999999997
    - type: recall_at_5
      value: 34.288000000000004
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
      value: 10.079
    - type: map_at_10
      value: 16.930999999999997
    - type: map_at_100
      value: 18.398999999999997
    - type: map_at_1000
      value: 18.561
    - type: map_at_3
      value: 14.294
    - type: map_at_5
      value: 15.579
    - type: mrr_at_1
      value: 22.606
    - type: mrr_at_10
      value: 32.513
    - type: mrr_at_100
      value: 33.463
    - type: mrr_at_1000
      value: 33.513999999999996
    - type: mrr_at_3
      value: 29.479
    - type: mrr_at_5
      value: 31.3
    - type: ndcg_at_1
      value: 22.606
    - type: ndcg_at_10
      value: 24.053
    - type: ndcg_at_100
      value: 30.258000000000003
    - type: ndcg_at_1000
      value: 33.516
    - type: ndcg_at_3
      value: 19.721
    - type: ndcg_at_5
      value: 21.144
    - type: precision_at_1
      value: 22.606
    - type: precision_at_10
      value: 7.55
    - type: precision_at_100
      value: 1.399
    - type: precision_at_1000
      value: 0.2
    - type: precision_at_3
      value: 14.701
    - type: precision_at_5
      value: 11.192
    - type: recall_at_1
      value: 10.079
    - type: recall_at_10
      value: 28.970000000000002
    - type: recall_at_100
      value: 50.805
    - type: recall_at_1000
      value: 69.378
    - type: recall_at_3
      value: 18.199
    - type: recall_at_5
      value: 22.442
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
      value: 7.794
    - type: map_at_10
      value: 15.165999999999999
    - type: map_at_100
      value: 20.508000000000003
    - type: map_at_1000
      value: 21.809
    - type: map_at_3
      value: 11.568000000000001
    - type: map_at_5
      value: 13.059000000000001
    - type: mrr_at_1
      value: 56.49999999999999
    - type: mrr_at_10
      value: 65.90899999999999
    - type: mrr_at_100
      value: 66.352
    - type: mrr_at_1000
      value: 66.369
    - type: mrr_at_3
      value: 64.0
    - type: mrr_at_5
      value: 65.10000000000001
    - type: ndcg_at_1
      value: 44.25
    - type: ndcg_at_10
      value: 32.649
    - type: ndcg_at_100
      value: 36.668
    - type: ndcg_at_1000
      value: 43.918
    - type: ndcg_at_3
      value: 37.096000000000004
    - type: ndcg_at_5
      value: 34.048
    - type: precision_at_1
      value: 56.49999999999999
    - type: precision_at_10
      value: 25.45
    - type: precision_at_100
      value: 8.055
    - type: precision_at_1000
      value: 1.7489999999999999
    - type: precision_at_3
      value: 41.0
    - type: precision_at_5
      value: 32.85
    - type: recall_at_1
      value: 7.794
    - type: recall_at_10
      value: 20.101
    - type: recall_at_100
      value: 42.448
    - type: recall_at_1000
      value: 65.88000000000001
    - type: recall_at_3
      value: 12.753
    - type: recall_at_5
      value: 15.307
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
      value: 44.01
    - type: f1
      value: 38.659680951114964
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
      value: 49.713
    - type: map_at_10
      value: 61.79
    - type: map_at_100
      value: 62.28
    - type: map_at_1000
      value: 62.297000000000004
    - type: map_at_3
      value: 59.361
    - type: map_at_5
      value: 60.92100000000001
    - type: mrr_at_1
      value: 53.405
    - type: mrr_at_10
      value: 65.79899999999999
    - type: mrr_at_100
      value: 66.219
    - type: mrr_at_1000
      value: 66.227
    - type: mrr_at_3
      value: 63.431000000000004
    - type: mrr_at_5
      value: 64.98
    - type: ndcg_at_1
      value: 53.405
    - type: ndcg_at_10
      value: 68.01899999999999
    - type: ndcg_at_100
      value: 70.197
    - type: ndcg_at_1000
      value: 70.571
    - type: ndcg_at_3
      value: 63.352
    - type: ndcg_at_5
      value: 66.018
    - type: precision_at_1
      value: 53.405
    - type: precision_at_10
      value: 9.119
    - type: precision_at_100
      value: 1.03
    - type: precision_at_1000
      value: 0.107
    - type: precision_at_3
      value: 25.602999999999998
    - type: precision_at_5
      value: 16.835
    - type: recall_at_1
      value: 49.713
    - type: recall_at_10
      value: 83.306
    - type: recall_at_100
      value: 92.92
    - type: recall_at_1000
      value: 95.577
    - type: recall_at_3
      value: 70.798
    - type: recall_at_5
      value: 77.254
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
      value: 15.310000000000002
    - type: map_at_10
      value: 26.204
    - type: map_at_100
      value: 27.932000000000002
    - type: map_at_1000
      value: 28.121000000000002
    - type: map_at_3
      value: 22.481
    - type: map_at_5
      value: 24.678
    - type: mrr_at_1
      value: 29.784
    - type: mrr_at_10
      value: 39.582
    - type: mrr_at_100
      value: 40.52
    - type: mrr_at_1000
      value: 40.568
    - type: mrr_at_3
      value: 37.114000000000004
    - type: mrr_at_5
      value: 38.596000000000004
    - type: ndcg_at_1
      value: 29.784
    - type: ndcg_at_10
      value: 33.432
    - type: ndcg_at_100
      value: 40.281
    - type: ndcg_at_1000
      value: 43.653999999999996
    - type: ndcg_at_3
      value: 29.612
    - type: ndcg_at_5
      value: 31.223
    - type: precision_at_1
      value: 29.784
    - type: precision_at_10
      value: 9.645
    - type: precision_at_100
      value: 1.645
    - type: precision_at_1000
      value: 0.22499999999999998
    - type: precision_at_3
      value: 20.165
    - type: precision_at_5
      value: 15.401000000000002
    - type: recall_at_1
      value: 15.310000000000002
    - type: recall_at_10
      value: 40.499
    - type: recall_at_100
      value: 66.643
    - type: recall_at_1000
      value: 87.059
    - type: recall_at_3
      value: 27.492
    - type: recall_at_5
      value: 33.748
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
      value: 33.599000000000004
    - type: map_at_10
      value: 47.347
    - type: map_at_100
      value: 48.191
    - type: map_at_1000
      value: 48.263
    - type: map_at_3
      value: 44.698
    - type: map_at_5
      value: 46.278999999999996
    - type: mrr_at_1
      value: 67.19800000000001
    - type: mrr_at_10
      value: 74.054
    - type: mrr_at_100
      value: 74.376
    - type: mrr_at_1000
      value: 74.392
    - type: mrr_at_3
      value: 72.849
    - type: mrr_at_5
      value: 73.643
    - type: ndcg_at_1
      value: 67.19800000000001
    - type: ndcg_at_10
      value: 56.482
    - type: ndcg_at_100
      value: 59.694
    - type: ndcg_at_1000
      value: 61.204
    - type: ndcg_at_3
      value: 52.43299999999999
    - type: ndcg_at_5
      value: 54.608000000000004
    - type: precision_at_1
      value: 67.19800000000001
    - type: precision_at_10
      value: 11.613999999999999
    - type: precision_at_100
      value: 1.415
    - type: precision_at_1000
      value: 0.16199999999999998
    - type: precision_at_3
      value: 32.726
    - type: precision_at_5
      value: 21.349999999999998
    - type: recall_at_1
      value: 33.599000000000004
    - type: recall_at_10
      value: 58.069
    - type: recall_at_100
      value: 70.736
    - type: recall_at_1000
      value: 80.804
    - type: recall_at_3
      value: 49.088
    - type: recall_at_5
      value: 53.376000000000005
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
      value: 73.64359999999999
    - type: ap
      value: 67.54685976014599
    - type: f1
      value: 73.55148707559482
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
      value: 19.502
    - type: map_at_10
      value: 30.816
    - type: map_at_100
      value: 32.007999999999996
    - type: map_at_1000
      value: 32.067
    - type: map_at_3
      value: 27.215
    - type: map_at_5
      value: 29.304000000000002
    - type: mrr_at_1
      value: 20.072000000000003
    - type: mrr_at_10
      value: 31.406
    - type: mrr_at_100
      value: 32.549
    - type: mrr_at_1000
      value: 32.602
    - type: mrr_at_3
      value: 27.839000000000002
    - type: mrr_at_5
      value: 29.926000000000002
    - type: ndcg_at_1
      value: 20.086000000000002
    - type: ndcg_at_10
      value: 37.282
    - type: ndcg_at_100
      value: 43.206
    - type: ndcg_at_1000
      value: 44.690000000000005
    - type: ndcg_at_3
      value: 29.932
    - type: ndcg_at_5
      value: 33.668
    - type: precision_at_1
      value: 20.086000000000002
    - type: precision_at_10
      value: 5.961
    - type: precision_at_100
      value: 0.898
    - type: precision_at_1000
      value: 0.10200000000000001
    - type: precision_at_3
      value: 12.856000000000002
    - type: precision_at_5
      value: 9.596
    - type: recall_at_1
      value: 19.502
    - type: recall_at_10
      value: 57.182
    - type: recall_at_100
      value: 84.952
    - type: recall_at_1000
      value: 96.34700000000001
    - type: recall_at_3
      value: 37.193
    - type: recall_at_5
      value: 46.157
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
      value: 93.96488828089375
    - type: f1
      value: 93.32119260543482
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
      value: 72.4965800273598
    - type: f1
      value: 49.34896217536082
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
      value: 67.60928043039678
    - type: f1
      value: 64.34244712074538
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
      value: 69.75453934095493
    - type: f1
      value: 68.39224867489249
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
      value: 31.862573504920082
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
      value: 27.511123551196803
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
      value: 30.99145104942086
    - type: mrr
      value: 32.03606480418627
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
      value: 5.015
    - type: map_at_10
      value: 11.054
    - type: map_at_100
      value: 13.773
    - type: map_at_1000
      value: 15.082999999999998
    - type: map_at_3
      value: 8.253
    - type: map_at_5
      value: 9.508999999999999
    - type: mrr_at_1
      value: 42.105
    - type: mrr_at_10
      value: 50.44499999999999
    - type: mrr_at_100
      value: 51.080000000000005
    - type: mrr_at_1000
      value: 51.129999999999995
    - type: mrr_at_3
      value: 48.555
    - type: mrr_at_5
      value: 49.84
    - type: ndcg_at_1
      value: 40.402
    - type: ndcg_at_10
      value: 30.403000000000002
    - type: ndcg_at_100
      value: 28.216
    - type: ndcg_at_1000
      value: 37.021
    - type: ndcg_at_3
      value: 35.53
    - type: ndcg_at_5
      value: 33.202999999999996
    - type: precision_at_1
      value: 42.105
    - type: precision_at_10
      value: 22.353
    - type: precision_at_100
      value: 7.266
    - type: precision_at_1000
      value: 2.011
    - type: precision_at_3
      value: 32.921
    - type: precision_at_5
      value: 28.297
    - type: recall_at_1
      value: 5.015
    - type: recall_at_10
      value: 14.393
    - type: recall_at_100
      value: 28.893
    - type: recall_at_1000
      value: 60.18
    - type: recall_at_3
      value: 9.184000000000001
    - type: recall_at_5
      value: 11.39
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
      value: 29.524
    - type: map_at_10
      value: 44.182
    - type: map_at_100
      value: 45.228
    - type: map_at_1000
      value: 45.265
    - type: map_at_3
      value: 39.978
    - type: map_at_5
      value: 42.482
    - type: mrr_at_1
      value: 33.256
    - type: mrr_at_10
      value: 46.661
    - type: mrr_at_100
      value: 47.47
    - type: mrr_at_1000
      value: 47.496
    - type: mrr_at_3
      value: 43.187999999999995
    - type: mrr_at_5
      value: 45.330999999999996
    - type: ndcg_at_1
      value: 33.227000000000004
    - type: ndcg_at_10
      value: 51.589
    - type: ndcg_at_100
      value: 56.043
    - type: ndcg_at_1000
      value: 56.937000000000005
    - type: ndcg_at_3
      value: 43.751
    - type: ndcg_at_5
      value: 47.937000000000005
    - type: precision_at_1
      value: 33.227000000000004
    - type: precision_at_10
      value: 8.556999999999999
    - type: precision_at_100
      value: 1.103
    - type: precision_at_1000
      value: 0.11900000000000001
    - type: precision_at_3
      value: 19.921
    - type: precision_at_5
      value: 14.396999999999998
    - type: recall_at_1
      value: 29.524
    - type: recall_at_10
      value: 71.615
    - type: recall_at_100
      value: 91.056
    - type: recall_at_1000
      value: 97.72800000000001
    - type: recall_at_3
      value: 51.451
    - type: recall_at_5
      value: 61.119
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
      value: 69.596
    - type: map_at_10
      value: 83.281
    - type: map_at_100
      value: 83.952
    - type: map_at_1000
      value: 83.97200000000001
    - type: map_at_3
      value: 80.315
    - type: map_at_5
      value: 82.223
    - type: mrr_at_1
      value: 80.17
    - type: mrr_at_10
      value: 86.522
    - type: mrr_at_100
      value: 86.644
    - type: mrr_at_1000
      value: 86.64500000000001
    - type: mrr_at_3
      value: 85.438
    - type: mrr_at_5
      value: 86.21799999999999
    - type: ndcg_at_1
      value: 80.19
    - type: ndcg_at_10
      value: 87.19
    - type: ndcg_at_100
      value: 88.567
    - type: ndcg_at_1000
      value: 88.70400000000001
    - type: ndcg_at_3
      value: 84.17999999999999
    - type: ndcg_at_5
      value: 85.931
    - type: precision_at_1
      value: 80.19
    - type: precision_at_10
      value: 13.209000000000001
    - type: precision_at_100
      value: 1.518
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 36.717
    - type: precision_at_5
      value: 24.248
    - type: recall_at_1
      value: 69.596
    - type: recall_at_10
      value: 94.533
    - type: recall_at_100
      value: 99.322
    - type: recall_at_1000
      value: 99.965
    - type: recall_at_3
      value: 85.911
    - type: recall_at_5
      value: 90.809
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
      value: 49.27650627571912
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
      value: 57.08550946534183
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
      value: 4.568
    - type: map_at_10
      value: 10.862
    - type: map_at_100
      value: 12.757
    - type: map_at_1000
      value: 13.031
    - type: map_at_3
      value: 7.960000000000001
    - type: map_at_5
      value: 9.337
    - type: mrr_at_1
      value: 22.5
    - type: mrr_at_10
      value: 32.6
    - type: mrr_at_100
      value: 33.603
    - type: mrr_at_1000
      value: 33.672000000000004
    - type: mrr_at_3
      value: 29.299999999999997
    - type: mrr_at_5
      value: 31.25
    - type: ndcg_at_1
      value: 22.5
    - type: ndcg_at_10
      value: 18.605
    - type: ndcg_at_100
      value: 26.029999999999998
    - type: ndcg_at_1000
      value: 31.256
    - type: ndcg_at_3
      value: 17.873
    - type: ndcg_at_5
      value: 15.511
    - type: precision_at_1
      value: 22.5
    - type: precision_at_10
      value: 9.58
    - type: precision_at_100
      value: 2.033
    - type: precision_at_1000
      value: 0.33
    - type: precision_at_3
      value: 16.633
    - type: precision_at_5
      value: 13.54
    - type: recall_at_1
      value: 4.568
    - type: recall_at_10
      value: 19.402
    - type: recall_at_100
      value: 41.277
    - type: recall_at_1000
      value: 66.963
    - type: recall_at_3
      value: 10.112
    - type: recall_at_5
      value: 13.712
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
      value: 83.31992291680787
    - type: cos_sim_spearman
      value: 76.7212346922664
    - type: euclidean_pearson
      value: 80.42189271706478
    - type: euclidean_spearman
      value: 76.7212342532493
    - type: manhattan_pearson
      value: 80.33171093031578
    - type: manhattan_spearman
      value: 76.63192883074694
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
      value: 83.16654278886763
    - type: cos_sim_spearman
      value: 73.66390263429565
    - type: euclidean_pearson
      value: 79.7485360086639
    - type: euclidean_spearman
      value: 73.66389870373436
    - type: manhattan_pearson
      value: 79.73652237443706
    - type: manhattan_spearman
      value: 73.65296117151647
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
      value: 82.40389689929246
    - type: cos_sim_spearman
      value: 83.29727595993955
    - type: euclidean_pearson
      value: 82.23970587854079
    - type: euclidean_spearman
      value: 83.29727595993955
    - type: manhattan_pearson
      value: 82.18823600831897
    - type: manhattan_spearman
      value: 83.20746192209594
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
      value: 81.73505246913413
    - type: cos_sim_spearman
      value: 79.1686548248754
    - type: euclidean_pearson
      value: 80.48889135993412
    - type: euclidean_spearman
      value: 79.16864112930354
    - type: manhattan_pearson
      value: 80.40720651057302
    - type: manhattan_spearman
      value: 79.0640155089286
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
      value: 86.3953512879065
    - type: cos_sim_spearman
      value: 87.29947322714338
    - type: euclidean_pearson
      value: 86.59759438529645
    - type: euclidean_spearman
      value: 87.29947511092824
    - type: manhattan_pearson
      value: 86.52097806169155
    - type: manhattan_spearman
      value: 87.22987242146534
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
      value: 82.48565753792056
    - type: cos_sim_spearman
      value: 83.6049720319893
    - type: euclidean_pearson
      value: 82.56452023172913
    - type: euclidean_spearman
      value: 83.60490168191697
    - type: manhattan_pearson
      value: 82.58079941137872
    - type: manhattan_spearman
      value: 83.60975807374051
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
      value: 88.18239976618212
    - type: cos_sim_spearman
      value: 88.23061724730616
    - type: euclidean_pearson
      value: 87.78482472776658
    - type: euclidean_spearman
      value: 88.23061724730616
    - type: manhattan_pearson
      value: 87.75059641730239
    - type: manhattan_spearman
      value: 88.22527413524622
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
      value: 63.42816418706765
    - type: cos_sim_spearman
      value: 63.4569864520124
    - type: euclidean_pearson
      value: 64.35405409953853
    - type: euclidean_spearman
      value: 63.4569864520124
    - type: manhattan_pearson
      value: 63.96649236073056
    - type: manhattan_spearman
      value: 63.01448583722708
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
      value: 83.41659638047614
    - type: cos_sim_spearman
      value: 84.03893866106175
    - type: euclidean_pearson
      value: 84.2251203953798
    - type: euclidean_spearman
      value: 84.03893866106175
    - type: manhattan_pearson
      value: 84.22733643205514
    - type: manhattan_spearman
      value: 84.06504411263612
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
      value: 79.75608022582414
    - type: mrr
      value: 94.0947732369301
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
      value: 50.161
    - type: map_at_10
      value: 59.458999999999996
    - type: map_at_100
      value: 60.156
    - type: map_at_1000
      value: 60.194
    - type: map_at_3
      value: 56.45400000000001
    - type: map_at_5
      value: 58.165
    - type: mrr_at_1
      value: 53.333
    - type: mrr_at_10
      value: 61.050000000000004
    - type: mrr_at_100
      value: 61.586
    - type: mrr_at_1000
      value: 61.624
    - type: mrr_at_3
      value: 58.889
    - type: mrr_at_5
      value: 60.122
    - type: ndcg_at_1
      value: 53.333
    - type: ndcg_at_10
      value: 63.888999999999996
    - type: ndcg_at_100
      value: 66.963
    - type: ndcg_at_1000
      value: 68.062
    - type: ndcg_at_3
      value: 59.01
    - type: ndcg_at_5
      value: 61.373999999999995
    - type: precision_at_1
      value: 53.333
    - type: precision_at_10
      value: 8.633000000000001
    - type: precision_at_100
      value: 1.027
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 23.111
    - type: precision_at_5
      value: 15.467
    - type: recall_at_1
      value: 50.161
    - type: recall_at_10
      value: 75.922
    - type: recall_at_100
      value: 90.0
    - type: recall_at_1000
      value: 98.667
    - type: recall_at_3
      value: 62.90599999999999
    - type: recall_at_5
      value: 68.828
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
      value: 99.81188118811882
    - type: cos_sim_ap
      value: 95.11619225962413
    - type: cos_sim_f1
      value: 90.35840484603736
    - type: cos_sim_precision
      value: 91.23343527013252
    - type: cos_sim_recall
      value: 89.5
    - type: dot_accuracy
      value: 99.81188118811882
    - type: dot_ap
      value: 95.11619225962413
    - type: dot_f1
      value: 90.35840484603736
    - type: dot_precision
      value: 91.23343527013252
    - type: dot_recall
      value: 89.5
    - type: euclidean_accuracy
      value: 99.81188118811882
    - type: euclidean_ap
      value: 95.11619225962413
    - type: euclidean_f1
      value: 90.35840484603736
    - type: euclidean_precision
      value: 91.23343527013252
    - type: euclidean_recall
      value: 89.5
    - type: manhattan_accuracy
      value: 99.80891089108911
    - type: manhattan_ap
      value: 95.07294266220966
    - type: manhattan_f1
      value: 90.21794221996959
    - type: manhattan_precision
      value: 91.46968139773895
    - type: manhattan_recall
      value: 89.0
    - type: max_accuracy
      value: 99.81188118811882
    - type: max_ap
      value: 95.11619225962413
    - type: max_f1
      value: 90.35840484603736
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
      value: 55.3481874105239
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
      value: 34.421291695525
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
      value: 49.98746633276634
    - type: mrr
      value: 50.63143249724133
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
      value: 31.009961979844036
    - type: cos_sim_spearman
      value: 30.558416108881044
    - type: dot_pearson
      value: 31.009964941134253
    - type: dot_spearman
      value: 30.545760761761393
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
      value: 0.207
    - type: map_at_10
      value: 1.6
    - type: map_at_100
      value: 8.594
    - type: map_at_1000
      value: 20.213
    - type: map_at_3
      value: 0.585
    - type: map_at_5
      value: 0.9039999999999999
    - type: mrr_at_1
      value: 78.0
    - type: mrr_at_10
      value: 87.4
    - type: mrr_at_100
      value: 87.4
    - type: mrr_at_1000
      value: 87.4
    - type: mrr_at_3
      value: 86.667
    - type: mrr_at_5
      value: 87.06700000000001
    - type: ndcg_at_1
      value: 73.0
    - type: ndcg_at_10
      value: 65.18
    - type: ndcg_at_100
      value: 49.631
    - type: ndcg_at_1000
      value: 43.498999999999995
    - type: ndcg_at_3
      value: 71.83800000000001
    - type: ndcg_at_5
      value: 69.271
    - type: precision_at_1
      value: 78.0
    - type: precision_at_10
      value: 69.19999999999999
    - type: precision_at_100
      value: 50.980000000000004
    - type: precision_at_1000
      value: 19.426
    - type: precision_at_3
      value: 77.333
    - type: precision_at_5
      value: 74.0
    - type: recall_at_1
      value: 0.207
    - type: recall_at_10
      value: 1.822
    - type: recall_at_100
      value: 11.849
    - type: recall_at_1000
      value: 40.492
    - type: recall_at_3
      value: 0.622
    - type: recall_at_5
      value: 0.9809999999999999
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
      value: 2.001
    - type: map_at_10
      value: 10.376000000000001
    - type: map_at_100
      value: 16.936999999999998
    - type: map_at_1000
      value: 18.615000000000002
    - type: map_at_3
      value: 5.335999999999999
    - type: map_at_5
      value: 7.374
    - type: mrr_at_1
      value: 20.408
    - type: mrr_at_10
      value: 38.29
    - type: mrr_at_100
      value: 39.33
    - type: mrr_at_1000
      value: 39.347
    - type: mrr_at_3
      value: 32.993
    - type: mrr_at_5
      value: 36.973
    - type: ndcg_at_1
      value: 17.347
    - type: ndcg_at_10
      value: 23.515
    - type: ndcg_at_100
      value: 37.457
    - type: ndcg_at_1000
      value: 49.439
    - type: ndcg_at_3
      value: 22.762999999999998
    - type: ndcg_at_5
      value: 22.622
    - type: precision_at_1
      value: 20.408
    - type: precision_at_10
      value: 22.448999999999998
    - type: precision_at_100
      value: 8.184
    - type: precision_at_1000
      value: 1.608
    - type: precision_at_3
      value: 25.85
    - type: precision_at_5
      value: 25.306
    - type: recall_at_1
      value: 2.001
    - type: recall_at_10
      value: 17.422
    - type: recall_at_100
      value: 51.532999999999994
    - type: recall_at_1000
      value: 87.466
    - type: recall_at_3
      value: 6.861000000000001
    - type: recall_at_5
      value: 10.502
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
      value: 71.54419999999999
    - type: ap
      value: 14.372170450843907
    - type: f1
      value: 54.94420257390529
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
      value: 59.402942840973395
    - type: f1
      value: 59.4166538875571
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
      value: 41.569064336457906
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
      value: 85.31322644096085
    - type: cos_sim_ap
      value: 72.14518894837381
    - type: cos_sim_f1
      value: 66.67489813557229
    - type: cos_sim_precision
      value: 62.65954977953121
    - type: cos_sim_recall
      value: 71.2401055408971
    - type: dot_accuracy
      value: 85.31322644096085
    - type: dot_ap
      value: 72.14521480685293
    - type: dot_f1
      value: 66.67489813557229
    - type: dot_precision
      value: 62.65954977953121
    - type: dot_recall
      value: 71.2401055408971
    - type: euclidean_accuracy
      value: 85.31322644096085
    - type: euclidean_ap
      value: 72.14520820485349
    - type: euclidean_f1
      value: 66.67489813557229
    - type: euclidean_precision
      value: 62.65954977953121
    - type: euclidean_recall
      value: 71.2401055408971
    - type: manhattan_accuracy
      value: 85.21785778148656
    - type: manhattan_ap
      value: 72.01177147657364
    - type: manhattan_f1
      value: 66.62594673833374
    - type: manhattan_precision
      value: 62.0336669699727
    - type: manhattan_recall
      value: 71.95250659630607
    - type: max_accuracy
      value: 85.31322644096085
    - type: max_ap
      value: 72.14521480685293
    - type: max_f1
      value: 66.67489813557229
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
      value: 89.12756626693057
    - type: cos_sim_ap
      value: 86.05430786440826
    - type: cos_sim_f1
      value: 78.27759692216631
    - type: cos_sim_precision
      value: 75.33466248931929
    - type: cos_sim_recall
      value: 81.45980905451185
    - type: dot_accuracy
      value: 89.12950673341872
    - type: dot_ap
      value: 86.05431161145492
    - type: dot_f1
      value: 78.27759692216631
    - type: dot_precision
      value: 75.33466248931929
    - type: dot_recall
      value: 81.45980905451185
    - type: euclidean_accuracy
      value: 89.12756626693057
    - type: euclidean_ap
      value: 86.05431303247397
    - type: euclidean_f1
      value: 78.27759692216631
    - type: euclidean_precision
      value: 75.33466248931929
    - type: euclidean_recall
      value: 81.45980905451185
    - type: manhattan_accuracy
      value: 89.04994760740482
    - type: manhattan_ap
      value: 86.00860610892074
    - type: manhattan_f1
      value: 78.1846776005392
    - type: manhattan_precision
      value: 76.10438839480975
    - type: manhattan_recall
      value: 80.3818909762858
    - type: max_accuracy
      value: 89.12950673341872
    - type: max_ap
      value: 86.05431303247397
    - type: max_f1
      value: 78.27759692216631
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

`jina-embeddings-v2-small-en` is an English, monolingual **embedding model** supporting **8192 sequence length**.
It is based on a Bert architecture (JinaBert) that supports the symmetric bidirectional variant of [ALiBi](https://arxiv.org/abs/2108.12409) to allow longer sequence length.
The backbone `jina-bert-v2-small-en` is pretrained on the C4 dataset.
The model is further trained on Jina AI's collection of more than 400 millions of sentence pairs and hard negatives.
These pairs were obtained from various domains and were carefully selected through a thorough cleaning process.

The embedding model was trained using 512 sequence length, but extrapolates to 8k sequence length (or even longer) thanks to ALiBi.
This makes our model useful for a range of use cases, especially when processing long documents is needed, including long document retrieval, semantic textual similarity, text reranking, recommendation, RAG and LLM-based generative search, etc.

This model has 33 million parameters, which enables lightning-fast and memory efficient inference, while still delivering impressive performance.
Additionally, we provide the following embedding models:

**V1 (Based on T5, 512 Seq)**

- [`jina-embeddings-v1-small-en`](https://huggingface.co/jinaai/jina-embedding-s-en-v1): 35 million parameters.
- [`jina-embeddings-v1-base-en`](https://huggingface.co/jinaai/jina-embedding-b-en-v1): 110 million parameters.
- [`jina-embeddings-v1-large-en`](https://huggingface.co/jinaai/jina-embedding-l-en-v1): 330 million parameters.

**V2 (Based on JinaBert, 8k Seq)**

- [`jina-embeddings-v2-small-en`](https://huggingface.co/jinaai/jina-embeddings-v2-small-en): 33 million parameters **(you are here)**.
- [`jina-embeddings-v2-base-en`](https://huggingface.co/jinaai/jina-embeddings-v2-base-en): 137 million parameters.
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
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-small-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
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

Alternatively, you can use Jina AI's [Embeddings platform](https://jina.ai/embeddings/) for fully-managed access to Jina Embeddings models.

## RAG Performance

Jina Embeddings are very effective for retrieval augmented generation (RAG).
Ravi Theja wrote a [blog post](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83) on using Jina Embeddings together with [LLama Index](https://github.com/run-llama/llama_index) for RAG:


<img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*ZP2RVejCZovF3FDCg-Bx3A.png" width="780px">

## Plans

The development of new bilingual models is currently underway. We will be targeting mainly the German and Spanish languages.
The upcoming models will be called `jina-embeddings-v2-small-de/es`.

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

<!---

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