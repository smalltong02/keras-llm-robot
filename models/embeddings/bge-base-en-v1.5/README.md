---
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
- mteb
model-index:
- name: bge-base-en-v1.5
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
      value: 76.14925373134328
    - type: ap
      value: 39.32336517995478
    - type: f1
      value: 70.16902252611425
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
      value: 93.386825
    - type: ap
      value: 90.21276917991995
    - type: f1
      value: 93.37741030006174
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
      value: 48.846000000000004
    - type: f1
      value: 48.14646269778261
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
      value: 40.754000000000005
    - type: map_at_10
      value: 55.761
    - type: map_at_100
      value: 56.330999999999996
    - type: map_at_1000
      value: 56.333999999999996
    - type: map_at_3
      value: 51.92
    - type: map_at_5
      value: 54.010999999999996
    - type: mrr_at_1
      value: 41.181
    - type: mrr_at_10
      value: 55.967999999999996
    - type: mrr_at_100
      value: 56.538
    - type: mrr_at_1000
      value: 56.542
    - type: mrr_at_3
      value: 51.980000000000004
    - type: mrr_at_5
      value: 54.208999999999996
    - type: ndcg_at_1
      value: 40.754000000000005
    - type: ndcg_at_10
      value: 63.605000000000004
    - type: ndcg_at_100
      value: 66.05199999999999
    - type: ndcg_at_1000
      value: 66.12
    - type: ndcg_at_3
      value: 55.708
    - type: ndcg_at_5
      value: 59.452000000000005
    - type: precision_at_1
      value: 40.754000000000005
    - type: precision_at_10
      value: 8.841000000000001
    - type: precision_at_100
      value: 0.991
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 22.238
    - type: precision_at_5
      value: 15.149000000000001
    - type: recall_at_1
      value: 40.754000000000005
    - type: recall_at_10
      value: 88.407
    - type: recall_at_100
      value: 99.14699999999999
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 66.714
    - type: recall_at_5
      value: 75.747
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
      value: 48.74884539679369
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
      value: 42.8075893810716
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
      value: 62.128470519187736
    - type: mrr
      value: 74.28065778481289
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
      value: 89.24629081484655
    - type: cos_sim_spearman
      value: 86.93752309911496
    - type: euclidean_pearson
      value: 87.58589628573816
    - type: euclidean_spearman
      value: 88.05622328825284
    - type: manhattan_pearson
      value: 87.5594959805773
    - type: manhattan_spearman
      value: 88.19658793233961
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
      value: 86.9512987012987
    - type: f1
      value: 86.92515357973708
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
      value: 39.10263762928872
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
      value: 36.69711517426737
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
      value: 32.327
    - type: map_at_10
      value: 44.099
    - type: map_at_100
      value: 45.525
    - type: map_at_1000
      value: 45.641999999999996
    - type: map_at_3
      value: 40.47
    - type: map_at_5
      value: 42.36
    - type: mrr_at_1
      value: 39.199
    - type: mrr_at_10
      value: 49.651
    - type: mrr_at_100
      value: 50.29
    - type: mrr_at_1000
      value: 50.329
    - type: mrr_at_3
      value: 46.924
    - type: mrr_at_5
      value: 48.548
    - type: ndcg_at_1
      value: 39.199
    - type: ndcg_at_10
      value: 50.773
    - type: ndcg_at_100
      value: 55.67999999999999
    - type: ndcg_at_1000
      value: 57.495
    - type: ndcg_at_3
      value: 45.513999999999996
    - type: ndcg_at_5
      value: 47.703
    - type: precision_at_1
      value: 39.199
    - type: precision_at_10
      value: 9.914000000000001
    - type: precision_at_100
      value: 1.5310000000000001
    - type: precision_at_1000
      value: 0.198
    - type: precision_at_3
      value: 21.984
    - type: precision_at_5
      value: 15.737000000000002
    - type: recall_at_1
      value: 32.327
    - type: recall_at_10
      value: 63.743
    - type: recall_at_100
      value: 84.538
    - type: recall_at_1000
      value: 96.089
    - type: recall_at_3
      value: 48.065000000000005
    - type: recall_at_5
      value: 54.519
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
      value: 32.671
    - type: map_at_10
      value: 42.954
    - type: map_at_100
      value: 44.151
    - type: map_at_1000
      value: 44.287
    - type: map_at_3
      value: 39.912
    - type: map_at_5
      value: 41.798
    - type: mrr_at_1
      value: 41.465
    - type: mrr_at_10
      value: 49.351
    - type: mrr_at_100
      value: 49.980000000000004
    - type: mrr_at_1000
      value: 50.016000000000005
    - type: mrr_at_3
      value: 47.144000000000005
    - type: mrr_at_5
      value: 48.592999999999996
    - type: ndcg_at_1
      value: 41.465
    - type: ndcg_at_10
      value: 48.565999999999995
    - type: ndcg_at_100
      value: 52.76499999999999
    - type: ndcg_at_1000
      value: 54.749
    - type: ndcg_at_3
      value: 44.57
    - type: ndcg_at_5
      value: 46.759
    - type: precision_at_1
      value: 41.465
    - type: precision_at_10
      value: 9.107999999999999
    - type: precision_at_100
      value: 1.433
    - type: precision_at_1000
      value: 0.191
    - type: precision_at_3
      value: 21.423000000000002
    - type: precision_at_5
      value: 15.414
    - type: recall_at_1
      value: 32.671
    - type: recall_at_10
      value: 57.738
    - type: recall_at_100
      value: 75.86500000000001
    - type: recall_at_1000
      value: 88.36
    - type: recall_at_3
      value: 45.626
    - type: recall_at_5
      value: 51.812000000000005
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
      value: 41.185
    - type: map_at_10
      value: 53.929
    - type: map_at_100
      value: 54.92
    - type: map_at_1000
      value: 54.967999999999996
    - type: map_at_3
      value: 50.70400000000001
    - type: map_at_5
      value: 52.673
    - type: mrr_at_1
      value: 47.398
    - type: mrr_at_10
      value: 57.303000000000004
    - type: mrr_at_100
      value: 57.959
    - type: mrr_at_1000
      value: 57.985
    - type: mrr_at_3
      value: 54.932
    - type: mrr_at_5
      value: 56.464999999999996
    - type: ndcg_at_1
      value: 47.398
    - type: ndcg_at_10
      value: 59.653
    - type: ndcg_at_100
      value: 63.627
    - type: ndcg_at_1000
      value: 64.596
    - type: ndcg_at_3
      value: 54.455
    - type: ndcg_at_5
      value: 57.245000000000005
    - type: precision_at_1
      value: 47.398
    - type: precision_at_10
      value: 9.524000000000001
    - type: precision_at_100
      value: 1.243
    - type: precision_at_1000
      value: 0.13699999999999998
    - type: precision_at_3
      value: 24.389
    - type: precision_at_5
      value: 16.752
    - type: recall_at_1
      value: 41.185
    - type: recall_at_10
      value: 73.193
    - type: recall_at_100
      value: 90.357
    - type: recall_at_1000
      value: 97.253
    - type: recall_at_3
      value: 59.199999999999996
    - type: recall_at_5
      value: 66.118
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
      value: 27.27
    - type: map_at_10
      value: 36.223
    - type: map_at_100
      value: 37.218
    - type: map_at_1000
      value: 37.293
    - type: map_at_3
      value: 33.503
    - type: map_at_5
      value: 35.097
    - type: mrr_at_1
      value: 29.492
    - type: mrr_at_10
      value: 38.352000000000004
    - type: mrr_at_100
      value: 39.188
    - type: mrr_at_1000
      value: 39.247
    - type: mrr_at_3
      value: 35.876000000000005
    - type: mrr_at_5
      value: 37.401
    - type: ndcg_at_1
      value: 29.492
    - type: ndcg_at_10
      value: 41.239
    - type: ndcg_at_100
      value: 46.066
    - type: ndcg_at_1000
      value: 47.992000000000004
    - type: ndcg_at_3
      value: 36.11
    - type: ndcg_at_5
      value: 38.772
    - type: precision_at_1
      value: 29.492
    - type: precision_at_10
      value: 6.260000000000001
    - type: precision_at_100
      value: 0.914
    - type: precision_at_1000
      value: 0.11100000000000002
    - type: precision_at_3
      value: 15.104000000000001
    - type: precision_at_5
      value: 10.644
    - type: recall_at_1
      value: 27.27
    - type: recall_at_10
      value: 54.589
    - type: recall_at_100
      value: 76.70700000000001
    - type: recall_at_1000
      value: 91.158
    - type: recall_at_3
      value: 40.974
    - type: recall_at_5
      value: 47.327000000000005
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
      value: 17.848
    - type: map_at_10
      value: 26.207
    - type: map_at_100
      value: 27.478
    - type: map_at_1000
      value: 27.602
    - type: map_at_3
      value: 23.405
    - type: map_at_5
      value: 24.98
    - type: mrr_at_1
      value: 21.891
    - type: mrr_at_10
      value: 31.041999999999998
    - type: mrr_at_100
      value: 32.092
    - type: mrr_at_1000
      value: 32.151999999999994
    - type: mrr_at_3
      value: 28.358
    - type: mrr_at_5
      value: 29.969
    - type: ndcg_at_1
      value: 21.891
    - type: ndcg_at_10
      value: 31.585
    - type: ndcg_at_100
      value: 37.531
    - type: ndcg_at_1000
      value: 40.256
    - type: ndcg_at_3
      value: 26.508
    - type: ndcg_at_5
      value: 28.894
    - type: precision_at_1
      value: 21.891
    - type: precision_at_10
      value: 5.795999999999999
    - type: precision_at_100
      value: 0.9990000000000001
    - type: precision_at_1000
      value: 0.13799999999999998
    - type: precision_at_3
      value: 12.769
    - type: precision_at_5
      value: 9.279
    - type: recall_at_1
      value: 17.848
    - type: recall_at_10
      value: 43.452
    - type: recall_at_100
      value: 69.216
    - type: recall_at_1000
      value: 88.102
    - type: recall_at_3
      value: 29.18
    - type: recall_at_5
      value: 35.347
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
      value: 30.94
    - type: map_at_10
      value: 41.248000000000005
    - type: map_at_100
      value: 42.495
    - type: map_at_1000
      value: 42.602000000000004
    - type: map_at_3
      value: 37.939
    - type: map_at_5
      value: 39.924
    - type: mrr_at_1
      value: 37.824999999999996
    - type: mrr_at_10
      value: 47.041
    - type: mrr_at_100
      value: 47.83
    - type: mrr_at_1000
      value: 47.878
    - type: mrr_at_3
      value: 44.466
    - type: mrr_at_5
      value: 46.111999999999995
    - type: ndcg_at_1
      value: 37.824999999999996
    - type: ndcg_at_10
      value: 47.223
    - type: ndcg_at_100
      value: 52.394
    - type: ndcg_at_1000
      value: 54.432
    - type: ndcg_at_3
      value: 42.032000000000004
    - type: ndcg_at_5
      value: 44.772
    - type: precision_at_1
      value: 37.824999999999996
    - type: precision_at_10
      value: 8.393
    - type: precision_at_100
      value: 1.2890000000000001
    - type: precision_at_1000
      value: 0.164
    - type: precision_at_3
      value: 19.698
    - type: precision_at_5
      value: 14.013
    - type: recall_at_1
      value: 30.94
    - type: recall_at_10
      value: 59.316
    - type: recall_at_100
      value: 80.783
    - type: recall_at_1000
      value: 94.15400000000001
    - type: recall_at_3
      value: 44.712
    - type: recall_at_5
      value: 51.932
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
      value: 27.104
    - type: map_at_10
      value: 36.675999999999995
    - type: map_at_100
      value: 38.076
    - type: map_at_1000
      value: 38.189
    - type: map_at_3
      value: 33.733999999999995
    - type: map_at_5
      value: 35.287
    - type: mrr_at_1
      value: 33.904
    - type: mrr_at_10
      value: 42.55
    - type: mrr_at_100
      value: 43.434
    - type: mrr_at_1000
      value: 43.494
    - type: mrr_at_3
      value: 40.126
    - type: mrr_at_5
      value: 41.473
    - type: ndcg_at_1
      value: 33.904
    - type: ndcg_at_10
      value: 42.414
    - type: ndcg_at_100
      value: 48.203
    - type: ndcg_at_1000
      value: 50.437
    - type: ndcg_at_3
      value: 37.633
    - type: ndcg_at_5
      value: 39.67
    - type: precision_at_1
      value: 33.904
    - type: precision_at_10
      value: 7.82
    - type: precision_at_100
      value: 1.2409999999999999
    - type: precision_at_1000
      value: 0.159
    - type: precision_at_3
      value: 17.884
    - type: precision_at_5
      value: 12.648000000000001
    - type: recall_at_1
      value: 27.104
    - type: recall_at_10
      value: 53.563
    - type: recall_at_100
      value: 78.557
    - type: recall_at_1000
      value: 93.533
    - type: recall_at_3
      value: 39.92
    - type: recall_at_5
      value: 45.457
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
      value: 27.707749999999997
    - type: map_at_10
      value: 36.961
    - type: map_at_100
      value: 38.158833333333334
    - type: map_at_1000
      value: 38.270333333333326
    - type: map_at_3
      value: 34.07183333333334
    - type: map_at_5
      value: 35.69533333333334
    - type: mrr_at_1
      value: 32.81875
    - type: mrr_at_10
      value: 41.293
    - type: mrr_at_100
      value: 42.116499999999995
    - type: mrr_at_1000
      value: 42.170249999999996
    - type: mrr_at_3
      value: 38.83983333333333
    - type: mrr_at_5
      value: 40.29775
    - type: ndcg_at_1
      value: 32.81875
    - type: ndcg_at_10
      value: 42.355
    - type: ndcg_at_100
      value: 47.41374999999999
    - type: ndcg_at_1000
      value: 49.5805
    - type: ndcg_at_3
      value: 37.52825
    - type: ndcg_at_5
      value: 39.83266666666667
    - type: precision_at_1
      value: 32.81875
    - type: precision_at_10
      value: 7.382416666666666
    - type: precision_at_100
      value: 1.1640833333333334
    - type: precision_at_1000
      value: 0.15383333333333335
    - type: precision_at_3
      value: 17.134166666666665
    - type: precision_at_5
      value: 12.174833333333336
    - type: recall_at_1
      value: 27.707749999999997
    - type: recall_at_10
      value: 53.945
    - type: recall_at_100
      value: 76.191
    - type: recall_at_1000
      value: 91.101
    - type: recall_at_3
      value: 40.39083333333334
    - type: recall_at_5
      value: 46.40083333333333
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
      value: 26.482
    - type: map_at_10
      value: 33.201
    - type: map_at_100
      value: 34.107
    - type: map_at_1000
      value: 34.197
    - type: map_at_3
      value: 31.174000000000003
    - type: map_at_5
      value: 32.279
    - type: mrr_at_1
      value: 29.908
    - type: mrr_at_10
      value: 36.235
    - type: mrr_at_100
      value: 37.04
    - type: mrr_at_1000
      value: 37.105
    - type: mrr_at_3
      value: 34.355999999999995
    - type: mrr_at_5
      value: 35.382999999999996
    - type: ndcg_at_1
      value: 29.908
    - type: ndcg_at_10
      value: 37.325
    - type: ndcg_at_100
      value: 41.795
    - type: ndcg_at_1000
      value: 44.105
    - type: ndcg_at_3
      value: 33.555
    - type: ndcg_at_5
      value: 35.266999999999996
    - type: precision_at_1
      value: 29.908
    - type: precision_at_10
      value: 5.721
    - type: precision_at_100
      value: 0.8630000000000001
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 14.008000000000001
    - type: precision_at_5
      value: 9.754999999999999
    - type: recall_at_1
      value: 26.482
    - type: recall_at_10
      value: 47.072
    - type: recall_at_100
      value: 67.27
    - type: recall_at_1000
      value: 84.371
    - type: recall_at_3
      value: 36.65
    - type: recall_at_5
      value: 40.774
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
      value: 18.815
    - type: map_at_10
      value: 26.369999999999997
    - type: map_at_100
      value: 27.458
    - type: map_at_1000
      value: 27.588
    - type: map_at_3
      value: 23.990000000000002
    - type: map_at_5
      value: 25.345000000000002
    - type: mrr_at_1
      value: 22.953000000000003
    - type: mrr_at_10
      value: 30.342999999999996
    - type: mrr_at_100
      value: 31.241000000000003
    - type: mrr_at_1000
      value: 31.319000000000003
    - type: mrr_at_3
      value: 28.16
    - type: mrr_at_5
      value: 29.406
    - type: ndcg_at_1
      value: 22.953000000000003
    - type: ndcg_at_10
      value: 31.151
    - type: ndcg_at_100
      value: 36.309000000000005
    - type: ndcg_at_1000
      value: 39.227000000000004
    - type: ndcg_at_3
      value: 26.921
    - type: ndcg_at_5
      value: 28.938000000000002
    - type: precision_at_1
      value: 22.953000000000003
    - type: precision_at_10
      value: 5.602
    - type: precision_at_100
      value: 0.9530000000000001
    - type: precision_at_1000
      value: 0.13899999999999998
    - type: precision_at_3
      value: 12.606
    - type: precision_at_5
      value: 9.119
    - type: recall_at_1
      value: 18.815
    - type: recall_at_10
      value: 41.574
    - type: recall_at_100
      value: 64.84400000000001
    - type: recall_at_1000
      value: 85.406
    - type: recall_at_3
      value: 29.694
    - type: recall_at_5
      value: 34.935
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
      value: 27.840999999999998
    - type: map_at_10
      value: 36.797999999999995
    - type: map_at_100
      value: 37.993
    - type: map_at_1000
      value: 38.086999999999996
    - type: map_at_3
      value: 34.050999999999995
    - type: map_at_5
      value: 35.379
    - type: mrr_at_1
      value: 32.649
    - type: mrr_at_10
      value: 41.025
    - type: mrr_at_100
      value: 41.878
    - type: mrr_at_1000
      value: 41.929
    - type: mrr_at_3
      value: 38.573
    - type: mrr_at_5
      value: 39.715
    - type: ndcg_at_1
      value: 32.649
    - type: ndcg_at_10
      value: 42.142
    - type: ndcg_at_100
      value: 47.558
    - type: ndcg_at_1000
      value: 49.643
    - type: ndcg_at_3
      value: 37.12
    - type: ndcg_at_5
      value: 38.983000000000004
    - type: precision_at_1
      value: 32.649
    - type: precision_at_10
      value: 7.08
    - type: precision_at_100
      value: 1.1039999999999999
    - type: precision_at_1000
      value: 0.13899999999999998
    - type: precision_at_3
      value: 16.698
    - type: precision_at_5
      value: 11.511000000000001
    - type: recall_at_1
      value: 27.840999999999998
    - type: recall_at_10
      value: 54.245
    - type: recall_at_100
      value: 77.947
    - type: recall_at_1000
      value: 92.36999999999999
    - type: recall_at_3
      value: 40.146
    - type: recall_at_5
      value: 44.951
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
      value: 26.529000000000003
    - type: map_at_10
      value: 35.010000000000005
    - type: map_at_100
      value: 36.647
    - type: map_at_1000
      value: 36.857
    - type: map_at_3
      value: 31.968000000000004
    - type: map_at_5
      value: 33.554
    - type: mrr_at_1
      value: 31.818
    - type: mrr_at_10
      value: 39.550999999999995
    - type: mrr_at_100
      value: 40.54
    - type: mrr_at_1000
      value: 40.596
    - type: mrr_at_3
      value: 36.726
    - type: mrr_at_5
      value: 38.416
    - type: ndcg_at_1
      value: 31.818
    - type: ndcg_at_10
      value: 40.675
    - type: ndcg_at_100
      value: 46.548
    - type: ndcg_at_1000
      value: 49.126
    - type: ndcg_at_3
      value: 35.829
    - type: ndcg_at_5
      value: 38.0
    - type: precision_at_1
      value: 31.818
    - type: precision_at_10
      value: 7.826
    - type: precision_at_100
      value: 1.538
    - type: precision_at_1000
      value: 0.24
    - type: precision_at_3
      value: 16.601
    - type: precision_at_5
      value: 12.095
    - type: recall_at_1
      value: 26.529000000000003
    - type: recall_at_10
      value: 51.03
    - type: recall_at_100
      value: 77.556
    - type: recall_at_1000
      value: 93.804
    - type: recall_at_3
      value: 36.986000000000004
    - type: recall_at_5
      value: 43.096000000000004
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
      value: 23.480999999999998
    - type: map_at_10
      value: 30.817
    - type: map_at_100
      value: 31.838
    - type: map_at_1000
      value: 31.932
    - type: map_at_3
      value: 28.011999999999997
    - type: map_at_5
      value: 29.668
    - type: mrr_at_1
      value: 25.323
    - type: mrr_at_10
      value: 33.072
    - type: mrr_at_100
      value: 33.926
    - type: mrr_at_1000
      value: 33.993
    - type: mrr_at_3
      value: 30.436999999999998
    - type: mrr_at_5
      value: 32.092
    - type: ndcg_at_1
      value: 25.323
    - type: ndcg_at_10
      value: 35.514
    - type: ndcg_at_100
      value: 40.489000000000004
    - type: ndcg_at_1000
      value: 42.908
    - type: ndcg_at_3
      value: 30.092000000000002
    - type: ndcg_at_5
      value: 32.989000000000004
    - type: precision_at_1
      value: 25.323
    - type: precision_at_10
      value: 5.545
    - type: precision_at_100
      value: 0.861
    - type: precision_at_1000
      value: 0.117
    - type: precision_at_3
      value: 12.446
    - type: precision_at_5
      value: 9.131
    - type: recall_at_1
      value: 23.480999999999998
    - type: recall_at_10
      value: 47.825
    - type: recall_at_100
      value: 70.652
    - type: recall_at_1000
      value: 88.612
    - type: recall_at_3
      value: 33.537
    - type: recall_at_5
      value: 40.542
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
      value: 13.333999999999998
    - type: map_at_10
      value: 22.524
    - type: map_at_100
      value: 24.506
    - type: map_at_1000
      value: 24.715
    - type: map_at_3
      value: 19.022
    - type: map_at_5
      value: 20.693
    - type: mrr_at_1
      value: 29.186
    - type: mrr_at_10
      value: 41.22
    - type: mrr_at_100
      value: 42.16
    - type: mrr_at_1000
      value: 42.192
    - type: mrr_at_3
      value: 38.013000000000005
    - type: mrr_at_5
      value: 39.704
    - type: ndcg_at_1
      value: 29.186
    - type: ndcg_at_10
      value: 31.167
    - type: ndcg_at_100
      value: 38.879000000000005
    - type: ndcg_at_1000
      value: 42.376000000000005
    - type: ndcg_at_3
      value: 25.817
    - type: ndcg_at_5
      value: 27.377000000000002
    - type: precision_at_1
      value: 29.186
    - type: precision_at_10
      value: 9.693999999999999
    - type: precision_at_100
      value: 1.8030000000000002
    - type: precision_at_1000
      value: 0.246
    - type: precision_at_3
      value: 19.11
    - type: precision_at_5
      value: 14.344999999999999
    - type: recall_at_1
      value: 13.333999999999998
    - type: recall_at_10
      value: 37.092000000000006
    - type: recall_at_100
      value: 63.651
    - type: recall_at_1000
      value: 83.05
    - type: recall_at_3
      value: 23.74
    - type: recall_at_5
      value: 28.655
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
      value: 9.151
    - type: map_at_10
      value: 19.653000000000002
    - type: map_at_100
      value: 28.053
    - type: map_at_1000
      value: 29.709000000000003
    - type: map_at_3
      value: 14.191
    - type: map_at_5
      value: 16.456
    - type: mrr_at_1
      value: 66.25
    - type: mrr_at_10
      value: 74.4
    - type: mrr_at_100
      value: 74.715
    - type: mrr_at_1000
      value: 74.726
    - type: mrr_at_3
      value: 72.417
    - type: mrr_at_5
      value: 73.667
    - type: ndcg_at_1
      value: 54.25
    - type: ndcg_at_10
      value: 40.77
    - type: ndcg_at_100
      value: 46.359
    - type: ndcg_at_1000
      value: 54.193000000000005
    - type: ndcg_at_3
      value: 44.832
    - type: ndcg_at_5
      value: 42.63
    - type: precision_at_1
      value: 66.25
    - type: precision_at_10
      value: 32.175
    - type: precision_at_100
      value: 10.668
    - type: precision_at_1000
      value: 2.067
    - type: precision_at_3
      value: 47.667
    - type: precision_at_5
      value: 41.3
    - type: recall_at_1
      value: 9.151
    - type: recall_at_10
      value: 25.003999999999998
    - type: recall_at_100
      value: 52.976
    - type: recall_at_1000
      value: 78.315
    - type: recall_at_3
      value: 15.487
    - type: recall_at_5
      value: 18.999
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
      value: 51.89999999999999
    - type: f1
      value: 46.47777925067403
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
      value: 73.706
    - type: map_at_10
      value: 82.423
    - type: map_at_100
      value: 82.67999999999999
    - type: map_at_1000
      value: 82.694
    - type: map_at_3
      value: 81.328
    - type: map_at_5
      value: 82.001
    - type: mrr_at_1
      value: 79.613
    - type: mrr_at_10
      value: 87.07000000000001
    - type: mrr_at_100
      value: 87.169
    - type: mrr_at_1000
      value: 87.17
    - type: mrr_at_3
      value: 86.404
    - type: mrr_at_5
      value: 86.856
    - type: ndcg_at_1
      value: 79.613
    - type: ndcg_at_10
      value: 86.289
    - type: ndcg_at_100
      value: 87.201
    - type: ndcg_at_1000
      value: 87.428
    - type: ndcg_at_3
      value: 84.625
    - type: ndcg_at_5
      value: 85.53699999999999
    - type: precision_at_1
      value: 79.613
    - type: precision_at_10
      value: 10.399
    - type: precision_at_100
      value: 1.1079999999999999
    - type: precision_at_1000
      value: 0.11499999999999999
    - type: precision_at_3
      value: 32.473
    - type: precision_at_5
      value: 20.132
    - type: recall_at_1
      value: 73.706
    - type: recall_at_10
      value: 93.559
    - type: recall_at_100
      value: 97.188
    - type: recall_at_1000
      value: 98.555
    - type: recall_at_3
      value: 88.98700000000001
    - type: recall_at_5
      value: 91.373
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
      value: 19.841
    - type: map_at_10
      value: 32.643
    - type: map_at_100
      value: 34.575
    - type: map_at_1000
      value: 34.736
    - type: map_at_3
      value: 28.317999999999998
    - type: map_at_5
      value: 30.964000000000002
    - type: mrr_at_1
      value: 39.660000000000004
    - type: mrr_at_10
      value: 48.620000000000005
    - type: mrr_at_100
      value: 49.384
    - type: mrr_at_1000
      value: 49.415
    - type: mrr_at_3
      value: 45.988
    - type: mrr_at_5
      value: 47.361
    - type: ndcg_at_1
      value: 39.660000000000004
    - type: ndcg_at_10
      value: 40.646
    - type: ndcg_at_100
      value: 47.657
    - type: ndcg_at_1000
      value: 50.428
    - type: ndcg_at_3
      value: 36.689
    - type: ndcg_at_5
      value: 38.211
    - type: precision_at_1
      value: 39.660000000000004
    - type: precision_at_10
      value: 11.235000000000001
    - type: precision_at_100
      value: 1.8530000000000002
    - type: precision_at_1000
      value: 0.23600000000000002
    - type: precision_at_3
      value: 24.587999999999997
    - type: precision_at_5
      value: 18.395
    - type: recall_at_1
      value: 19.841
    - type: recall_at_10
      value: 48.135
    - type: recall_at_100
      value: 74.224
    - type: recall_at_1000
      value: 90.826
    - type: recall_at_3
      value: 33.536
    - type: recall_at_5
      value: 40.311
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
      value: 40.358
    - type: map_at_10
      value: 64.497
    - type: map_at_100
      value: 65.362
    - type: map_at_1000
      value: 65.41900000000001
    - type: map_at_3
      value: 61.06700000000001
    - type: map_at_5
      value: 63.317
    - type: mrr_at_1
      value: 80.716
    - type: mrr_at_10
      value: 86.10799999999999
    - type: mrr_at_100
      value: 86.265
    - type: mrr_at_1000
      value: 86.27
    - type: mrr_at_3
      value: 85.271
    - type: mrr_at_5
      value: 85.82499999999999
    - type: ndcg_at_1
      value: 80.716
    - type: ndcg_at_10
      value: 72.597
    - type: ndcg_at_100
      value: 75.549
    - type: ndcg_at_1000
      value: 76.61
    - type: ndcg_at_3
      value: 67.874
    - type: ndcg_at_5
      value: 70.655
    - type: precision_at_1
      value: 80.716
    - type: precision_at_10
      value: 15.148
    - type: precision_at_100
      value: 1.745
    - type: precision_at_1000
      value: 0.188
    - type: precision_at_3
      value: 43.597
    - type: precision_at_5
      value: 28.351
    - type: recall_at_1
      value: 40.358
    - type: recall_at_10
      value: 75.739
    - type: recall_at_100
      value: 87.259
    - type: recall_at_1000
      value: 94.234
    - type: recall_at_3
      value: 65.39500000000001
    - type: recall_at_5
      value: 70.878
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
      value: 90.80799999999998
    - type: ap
      value: 86.81350378180757
    - type: f1
      value: 90.79901248314215
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
      value: 22.096
    - type: map_at_10
      value: 34.384
    - type: map_at_100
      value: 35.541
    - type: map_at_1000
      value: 35.589999999999996
    - type: map_at_3
      value: 30.496000000000002
    - type: map_at_5
      value: 32.718
    - type: mrr_at_1
      value: 22.750999999999998
    - type: mrr_at_10
      value: 35.024
    - type: mrr_at_100
      value: 36.125
    - type: mrr_at_1000
      value: 36.168
    - type: mrr_at_3
      value: 31.225
    - type: mrr_at_5
      value: 33.416000000000004
    - type: ndcg_at_1
      value: 22.750999999999998
    - type: ndcg_at_10
      value: 41.351
    - type: ndcg_at_100
      value: 46.92
    - type: ndcg_at_1000
      value: 48.111
    - type: ndcg_at_3
      value: 33.439
    - type: ndcg_at_5
      value: 37.407000000000004
    - type: precision_at_1
      value: 22.750999999999998
    - type: precision_at_10
      value: 6.564
    - type: precision_at_100
      value: 0.935
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 14.288
    - type: precision_at_5
      value: 10.581999999999999
    - type: recall_at_1
      value: 22.096
    - type: recall_at_10
      value: 62.771
    - type: recall_at_100
      value: 88.529
    - type: recall_at_1000
      value: 97.55
    - type: recall_at_3
      value: 41.245
    - type: recall_at_5
      value: 50.788
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
      value: 94.16780665754673
    - type: f1
      value: 93.96331194859894
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
      value: 76.90606475148198
    - type: f1
      value: 58.58344986604187
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
      value: 76.14660390047075
    - type: f1
      value: 74.31533923533614
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
      value: 80.16139878950908
    - type: f1
      value: 80.18532656824924
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
      value: 32.949880906135085
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
      value: 31.56300351524862
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
      value: 31.196521894371315
    - type: mrr
      value: 32.22644231694389
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
      value: 6.783
    - type: map_at_10
      value: 14.549000000000001
    - type: map_at_100
      value: 18.433
    - type: map_at_1000
      value: 19.949
    - type: map_at_3
      value: 10.936
    - type: map_at_5
      value: 12.514
    - type: mrr_at_1
      value: 47.368
    - type: mrr_at_10
      value: 56.42
    - type: mrr_at_100
      value: 56.908
    - type: mrr_at_1000
      value: 56.95
    - type: mrr_at_3
      value: 54.283
    - type: mrr_at_5
      value: 55.568
    - type: ndcg_at_1
      value: 45.666000000000004
    - type: ndcg_at_10
      value: 37.389
    - type: ndcg_at_100
      value: 34.253
    - type: ndcg_at_1000
      value: 43.059999999999995
    - type: ndcg_at_3
      value: 42.725
    - type: ndcg_at_5
      value: 40.193
    - type: precision_at_1
      value: 47.368
    - type: precision_at_10
      value: 27.988000000000003
    - type: precision_at_100
      value: 8.672
    - type: precision_at_1000
      value: 2.164
    - type: precision_at_3
      value: 40.248
    - type: precision_at_5
      value: 34.737
    - type: recall_at_1
      value: 6.783
    - type: recall_at_10
      value: 17.838
    - type: recall_at_100
      value: 33.672000000000004
    - type: recall_at_1000
      value: 66.166
    - type: recall_at_3
      value: 11.849
    - type: recall_at_5
      value: 14.205000000000002
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
      value: 31.698999999999998
    - type: map_at_10
      value: 46.556
    - type: map_at_100
      value: 47.652
    - type: map_at_1000
      value: 47.68
    - type: map_at_3
      value: 42.492000000000004
    - type: map_at_5
      value: 44.763999999999996
    - type: mrr_at_1
      value: 35.747
    - type: mrr_at_10
      value: 49.242999999999995
    - type: mrr_at_100
      value: 50.052
    - type: mrr_at_1000
      value: 50.068
    - type: mrr_at_3
      value: 45.867000000000004
    - type: mrr_at_5
      value: 47.778999999999996
    - type: ndcg_at_1
      value: 35.717999999999996
    - type: ndcg_at_10
      value: 54.14600000000001
    - type: ndcg_at_100
      value: 58.672999999999995
    - type: ndcg_at_1000
      value: 59.279
    - type: ndcg_at_3
      value: 46.407
    - type: ndcg_at_5
      value: 50.181
    - type: precision_at_1
      value: 35.717999999999996
    - type: precision_at_10
      value: 8.844000000000001
    - type: precision_at_100
      value: 1.139
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 20.993000000000002
    - type: precision_at_5
      value: 14.791000000000002
    - type: recall_at_1
      value: 31.698999999999998
    - type: recall_at_10
      value: 74.693
    - type: recall_at_100
      value: 94.15299999999999
    - type: recall_at_1000
      value: 98.585
    - type: recall_at_3
      value: 54.388999999999996
    - type: recall_at_5
      value: 63.08200000000001
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
      value: 71.283
    - type: map_at_10
      value: 85.24000000000001
    - type: map_at_100
      value: 85.882
    - type: map_at_1000
      value: 85.897
    - type: map_at_3
      value: 82.326
    - type: map_at_5
      value: 84.177
    - type: mrr_at_1
      value: 82.21000000000001
    - type: mrr_at_10
      value: 88.228
    - type: mrr_at_100
      value: 88.32
    - type: mrr_at_1000
      value: 88.32
    - type: mrr_at_3
      value: 87.323
    - type: mrr_at_5
      value: 87.94800000000001
    - type: ndcg_at_1
      value: 82.17999999999999
    - type: ndcg_at_10
      value: 88.9
    - type: ndcg_at_100
      value: 90.079
    - type: ndcg_at_1000
      value: 90.158
    - type: ndcg_at_3
      value: 86.18299999999999
    - type: ndcg_at_5
      value: 87.71799999999999
    - type: precision_at_1
      value: 82.17999999999999
    - type: precision_at_10
      value: 13.464
    - type: precision_at_100
      value: 1.533
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.693
    - type: precision_at_5
      value: 24.792
    - type: recall_at_1
      value: 71.283
    - type: recall_at_10
      value: 95.742
    - type: recall_at_100
      value: 99.67200000000001
    - type: recall_at_1000
      value: 99.981
    - type: recall_at_3
      value: 87.888
    - type: recall_at_5
      value: 92.24
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
      value: 56.24267063669042
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
      value: 62.88056988932578
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
      value: 4.903
    - type: map_at_10
      value: 13.202
    - type: map_at_100
      value: 15.5
    - type: map_at_1000
      value: 15.870999999999999
    - type: map_at_3
      value: 9.407
    - type: map_at_5
      value: 11.238
    - type: mrr_at_1
      value: 24.2
    - type: mrr_at_10
      value: 35.867
    - type: mrr_at_100
      value: 37.001
    - type: mrr_at_1000
      value: 37.043
    - type: mrr_at_3
      value: 32.5
    - type: mrr_at_5
      value: 34.35
    - type: ndcg_at_1
      value: 24.2
    - type: ndcg_at_10
      value: 21.731
    - type: ndcg_at_100
      value: 30.7
    - type: ndcg_at_1000
      value: 36.618
    - type: ndcg_at_3
      value: 20.72
    - type: ndcg_at_5
      value: 17.954
    - type: precision_at_1
      value: 24.2
    - type: precision_at_10
      value: 11.33
    - type: precision_at_100
      value: 2.4410000000000003
    - type: precision_at_1000
      value: 0.386
    - type: precision_at_3
      value: 19.667
    - type: precision_at_5
      value: 15.86
    - type: recall_at_1
      value: 4.903
    - type: recall_at_10
      value: 22.962
    - type: recall_at_100
      value: 49.563
    - type: recall_at_1000
      value: 78.238
    - type: recall_at_3
      value: 11.953
    - type: recall_at_5
      value: 16.067999999999998
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
      value: 84.12694254604078
    - type: cos_sim_spearman
      value: 80.30141815181918
    - type: euclidean_pearson
      value: 81.34015449877128
    - type: euclidean_spearman
      value: 80.13984197010849
    - type: manhattan_pearson
      value: 81.31767068124086
    - type: manhattan_spearman
      value: 80.11720513114103
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
      value: 86.13112984010417
    - type: cos_sim_spearman
      value: 78.03063573402875
    - type: euclidean_pearson
      value: 83.51928418844804
    - type: euclidean_spearman
      value: 78.4045235411144
    - type: manhattan_pearson
      value: 83.49981637388689
    - type: manhattan_spearman
      value: 78.4042575139372
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
      value: 82.50327987379504
    - type: cos_sim_spearman
      value: 84.18556767756205
    - type: euclidean_pearson
      value: 82.69684424327679
    - type: euclidean_spearman
      value: 83.5368106038335
    - type: manhattan_pearson
      value: 82.57967581007374
    - type: manhattan_spearman
      value: 83.43009053133697
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
      value: 82.50756863007814
    - type: cos_sim_spearman
      value: 82.27204331279108
    - type: euclidean_pearson
      value: 81.39535251429741
    - type: euclidean_spearman
      value: 81.84386626336239
    - type: manhattan_pearson
      value: 81.34281737280695
    - type: manhattan_spearman
      value: 81.81149375673166
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
      value: 86.8727714856726
    - type: cos_sim_spearman
      value: 87.95738287792312
    - type: euclidean_pearson
      value: 86.62920602795887
    - type: euclidean_spearman
      value: 87.05207355381243
    - type: manhattan_pearson
      value: 86.53587918472225
    - type: manhattan_spearman
      value: 86.95382961029586
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
      value: 83.52240359769479
    - type: cos_sim_spearman
      value: 85.47685776238286
    - type: euclidean_pearson
      value: 84.25815333483058
    - type: euclidean_spearman
      value: 85.27415639683198
    - type: manhattan_pearson
      value: 84.29127757025637
    - type: manhattan_spearman
      value: 85.30226224917351
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
      value: 86.42501708915708
    - type: cos_sim_spearman
      value: 86.42276182795041
    - type: euclidean_pearson
      value: 86.5408207354761
    - type: euclidean_spearman
      value: 85.46096321750838
    - type: manhattan_pearson
      value: 86.54177303026881
    - type: manhattan_spearman
      value: 85.50313151916117
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
      value: 64.86521089250766
    - type: cos_sim_spearman
      value: 65.94868540323003
    - type: euclidean_pearson
      value: 67.16569626533084
    - type: euclidean_spearman
      value: 66.37667004134917
    - type: manhattan_pearson
      value: 67.1482365102333
    - type: manhattan_spearman
      value: 66.53240122580029
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
      value: 84.64746265365318
    - type: cos_sim_spearman
      value: 86.41888825906786
    - type: euclidean_pearson
      value: 85.27453642725811
    - type: euclidean_spearman
      value: 85.94095796602544
    - type: manhattan_pearson
      value: 85.28643660505334
    - type: manhattan_spearman
      value: 85.95028003260744
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
      value: 87.48903153618527
    - type: mrr
      value: 96.41081503826601
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
      value: 58.594
    - type: map_at_10
      value: 69.296
    - type: map_at_100
      value: 69.782
    - type: map_at_1000
      value: 69.795
    - type: map_at_3
      value: 66.23
    - type: map_at_5
      value: 68.293
    - type: mrr_at_1
      value: 61.667
    - type: mrr_at_10
      value: 70.339
    - type: mrr_at_100
      value: 70.708
    - type: mrr_at_1000
      value: 70.722
    - type: mrr_at_3
      value: 68.0
    - type: mrr_at_5
      value: 69.56700000000001
    - type: ndcg_at_1
      value: 61.667
    - type: ndcg_at_10
      value: 74.039
    - type: ndcg_at_100
      value: 76.103
    - type: ndcg_at_1000
      value: 76.47800000000001
    - type: ndcg_at_3
      value: 68.967
    - type: ndcg_at_5
      value: 71.96900000000001
    - type: precision_at_1
      value: 61.667
    - type: precision_at_10
      value: 9.866999999999999
    - type: precision_at_100
      value: 1.097
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 27.111
    - type: precision_at_5
      value: 18.2
    - type: recall_at_1
      value: 58.594
    - type: recall_at_10
      value: 87.422
    - type: recall_at_100
      value: 96.667
    - type: recall_at_1000
      value: 99.667
    - type: recall_at_3
      value: 74.217
    - type: recall_at_5
      value: 81.539
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
      value: 99.85049504950496
    - type: cos_sim_ap
      value: 96.33111544137081
    - type: cos_sim_f1
      value: 92.35443037974684
    - type: cos_sim_precision
      value: 93.53846153846153
    - type: cos_sim_recall
      value: 91.2
    - type: dot_accuracy
      value: 99.82376237623762
    - type: dot_ap
      value: 95.38082527310888
    - type: dot_f1
      value: 90.90909090909092
    - type: dot_precision
      value: 92.90187891440502
    - type: dot_recall
      value: 89.0
    - type: euclidean_accuracy
      value: 99.84851485148515
    - type: euclidean_ap
      value: 96.32316003996347
    - type: euclidean_f1
      value: 92.2071392659628
    - type: euclidean_precision
      value: 92.71991911021233
    - type: euclidean_recall
      value: 91.7
    - type: manhattan_accuracy
      value: 99.84851485148515
    - type: manhattan_ap
      value: 96.3655668249217
    - type: manhattan_f1
      value: 92.18356026222895
    - type: manhattan_precision
      value: 92.98067141403867
    - type: manhattan_recall
      value: 91.4
    - type: max_accuracy
      value: 99.85049504950496
    - type: max_ap
      value: 96.3655668249217
    - type: max_f1
      value: 92.35443037974684
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
      value: 65.94861371629051
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
      value: 35.009430451385
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
      value: 54.61164066427969
    - type: mrr
      value: 55.49710603938544
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
      value: 30.622620124907662
    - type: cos_sim_spearman
      value: 31.0678351356163
    - type: dot_pearson
      value: 30.863727693306814
    - type: dot_spearman
      value: 31.230306567021255
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
      value: 0.22
    - type: map_at_10
      value: 2.011
    - type: map_at_100
      value: 10.974
    - type: map_at_1000
      value: 25.819
    - type: map_at_3
      value: 0.6649999999999999
    - type: map_at_5
      value: 1.076
    - type: mrr_at_1
      value: 86.0
    - type: mrr_at_10
      value: 91.8
    - type: mrr_at_100
      value: 91.8
    - type: mrr_at_1000
      value: 91.8
    - type: mrr_at_3
      value: 91.0
    - type: mrr_at_5
      value: 91.8
    - type: ndcg_at_1
      value: 82.0
    - type: ndcg_at_10
      value: 78.07300000000001
    - type: ndcg_at_100
      value: 58.231
    - type: ndcg_at_1000
      value: 51.153000000000006
    - type: ndcg_at_3
      value: 81.123
    - type: ndcg_at_5
      value: 81.059
    - type: precision_at_1
      value: 86.0
    - type: precision_at_10
      value: 83.0
    - type: precision_at_100
      value: 59.38
    - type: precision_at_1000
      value: 22.55
    - type: precision_at_3
      value: 87.333
    - type: precision_at_5
      value: 86.8
    - type: recall_at_1
      value: 0.22
    - type: recall_at_10
      value: 2.2079999999999997
    - type: recall_at_100
      value: 14.069
    - type: recall_at_1000
      value: 47.678
    - type: recall_at_3
      value: 0.7040000000000001
    - type: recall_at_5
      value: 1.161
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
      value: 2.809
    - type: map_at_10
      value: 10.394
    - type: map_at_100
      value: 16.598
    - type: map_at_1000
      value: 18.142
    - type: map_at_3
      value: 5.572
    - type: map_at_5
      value: 7.1370000000000005
    - type: mrr_at_1
      value: 32.653
    - type: mrr_at_10
      value: 46.564
    - type: mrr_at_100
      value: 47.469
    - type: mrr_at_1000
      value: 47.469
    - type: mrr_at_3
      value: 42.177
    - type: mrr_at_5
      value: 44.524
    - type: ndcg_at_1
      value: 30.612000000000002
    - type: ndcg_at_10
      value: 25.701
    - type: ndcg_at_100
      value: 37.532
    - type: ndcg_at_1000
      value: 48.757
    - type: ndcg_at_3
      value: 28.199999999999996
    - type: ndcg_at_5
      value: 25.987
    - type: precision_at_1
      value: 32.653
    - type: precision_at_10
      value: 23.469
    - type: precision_at_100
      value: 7.9799999999999995
    - type: precision_at_1000
      value: 1.5350000000000001
    - type: precision_at_3
      value: 29.932
    - type: precision_at_5
      value: 26.122
    - type: recall_at_1
      value: 2.809
    - type: recall_at_10
      value: 16.887
    - type: recall_at_100
      value: 48.67
    - type: recall_at_1000
      value: 82.89699999999999
    - type: recall_at_3
      value: 6.521000000000001
    - type: recall_at_5
      value: 9.609
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
      value: 71.57860000000001
    - type: ap
      value: 13.82629211536393
    - type: f1
      value: 54.59860966183956
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
      value: 59.38030560271647
    - type: f1
      value: 59.69685552567865
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
      value: 51.4736717043405
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
      value: 86.92853311080646
    - type: cos_sim_ap
      value: 77.67872502591382
    - type: cos_sim_f1
      value: 70.33941236068895
    - type: cos_sim_precision
      value: 67.63273258645884
    - type: cos_sim_recall
      value: 73.27176781002639
    - type: dot_accuracy
      value: 85.79603027954938
    - type: dot_ap
      value: 73.73786190233379
    - type: dot_f1
      value: 67.3437901774235
    - type: dot_precision
      value: 65.67201604814443
    - type: dot_recall
      value: 69.10290237467018
    - type: euclidean_accuracy
      value: 86.94045419324074
    - type: euclidean_ap
      value: 77.6687791535167
    - type: euclidean_f1
      value: 70.47209214023542
    - type: euclidean_precision
      value: 67.7207492094381
    - type: euclidean_recall
      value: 73.45646437994723
    - type: manhattan_accuracy
      value: 86.87488823985218
    - type: manhattan_ap
      value: 77.63373392430728
    - type: manhattan_f1
      value: 70.40920716112532
    - type: manhattan_precision
      value: 68.31265508684864
    - type: manhattan_recall
      value: 72.63852242744063
    - type: max_accuracy
      value: 86.94045419324074
    - type: max_ap
      value: 77.67872502591382
    - type: max_f1
      value: 70.47209214023542
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
      value: 88.67155664221679
    - type: cos_sim_ap
      value: 85.64591703003417
    - type: cos_sim_f1
      value: 77.59531005352656
    - type: cos_sim_precision
      value: 73.60967184801382
    - type: cos_sim_recall
      value: 82.03726516784724
    - type: dot_accuracy
      value: 88.41541506578181
    - type: dot_ap
      value: 84.6482788957769
    - type: dot_f1
      value: 77.04748541466657
    - type: dot_precision
      value: 74.02440754931176
    - type: dot_recall
      value: 80.3279950723745
    - type: euclidean_accuracy
      value: 88.63080684596576
    - type: euclidean_ap
      value: 85.44570045321562
    - type: euclidean_f1
      value: 77.28769403336106
    - type: euclidean_precision
      value: 72.90600040958427
    - type: euclidean_recall
      value: 82.22975053895904
    - type: manhattan_accuracy
      value: 88.59393798269105
    - type: manhattan_ap
      value: 85.40271361038187
    - type: manhattan_f1
      value: 77.17606419344392
    - type: manhattan_precision
      value: 72.4447747078295
    - type: manhattan_recall
      value: 82.5685247921158
    - type: max_accuracy
      value: 88.67155664221679
    - type: max_ap
      value: 85.64591703003417
    - type: max_f1
      value: 77.59531005352656
license: mit
language:
- en
---


<h1 align="center">FlagEmbedding</h1>


<h4 align="center">
    <p>
        <a href=#model-list>Model List</a> | 
        <a href=#frequently-asked-questions>FAQ</a> |
        <a href=#usage>Usage</a>  |
        <a href="#evaluation">Evaluation</a> |
        <a href="#train">Train</a> |
        <a href="#contact">Contact</a> |
        <a href="#citation">Citation</a> |
        <a href="#license">License</a> 
    <p>
</h4>

More details please refer to our Github: [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding).


[English](README.md) | [中文](https://github.com/FlagOpen/FlagEmbedding/blob/master/README_zh.md)

FlagEmbedding can map any text to a low-dimensional dense vector which can be used for tasks like retrieval, classification,  clustering, or semantic search.
And it also can be used in vector databases for LLMs.

************* 🌟**Updates**🌟 *************
- 10/12/2023: Release [LLM-Embedder](./FlagEmbedding/llm_embedder/README.md), a unified embedding model to support diverse retrieval augmentation needs for LLMs. [Paper](https://arxiv.org/pdf/2310.07554.pdf)  :fire:  
- 09/15/2023: The [technical report](https://arxiv.org/pdf/2309.07597.pdf) of BGE has been released 
- 09/15/2023: The [masive training data](https://data.baai.ac.cn/details/BAAI-MTP) of BGE has been released 
- 09/12/2023: New models: 
    - **New reranker model**: release cross-encoder models `BAAI/bge-reranker-base` and `BAAI/bge-reranker-large`, which are more powerful than embedding model. We recommend to use/fine-tune them to re-rank top-k documents returned by embedding models. 
    - **update embedding model**: release `bge-*-v1.5` embedding model to alleviate the issue of the similarity distribution, and enhance its retrieval ability without instruction.
 

<details>
  <summary>More</summary>
<!-- ### More -->
    
- 09/07/2023: Update [fine-tune code](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md): Add script to mine hard negatives and support adding instruction during fine-tuning. 
- 08/09/2023: BGE Models are integrated into **Langchain**, you can use it like [this](#using-langchain); C-MTEB **leaderboard** is [available](https://huggingface.co/spaces/mteb/leaderboard).  
- 08/05/2023: Release base-scale and small-scale models, **best performance among the models of the same size 🤗**  
- 08/02/2023: Release `bge-large-*`(short for BAAI General Embedding) Models, **rank 1st on MTEB and C-MTEB benchmark!** :tada: :tada:   
- 08/01/2023: We release the [Chinese Massive Text Embedding Benchmark](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB) (**C-MTEB**), consisting of 31 test dataset.  
  
</details>


## Model List

`bge` is short for `BAAI general embedding`.

|              Model              | Language | | Description | query instruction for retrieval [1] |
|:-------------------------------|:--------:| :--------:| :--------:|:--------:|
|  [BAAI/llm-embedder](https://huggingface.co/BAAI/llm-embedder)  |   English | [Inference](./FlagEmbedding/llm_embedder/README.md) [Fine-tune](./FlagEmbedding/llm_embedder/README.md) | a unified embedding model to support diverse retrieval augmentation needs for LLMs | See [README](./FlagEmbedding/llm_embedder/README.md) |
|  [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)  |   Chinese and English | [Inference](#usage-for-reranker) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) | a cross-encoder model which is more accurate but less efficient [2] |   |
|  [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) |   Chinese and English | [Inference](#usage-for-reranker) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker) | a cross-encoder model which is more accurate but less efficient [2] |   |
|  [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution  | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) |   Chinese |  [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | version 1.5 with more reasonable similarity distribution | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | :trophy: rank **1st** in [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | a base-scale model but with similar ability to `bge-large-en` | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-small-en](https://huggingface.co/BAAI/bge-small-en) |   English | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) |a small-scale model but with competitive performance  | `Represent this sentence for searching relevant passages: `  |
|  [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | :trophy: rank **1st** in [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB) benchmark | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) |   Chinese |  [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | a base-scale model but with similar ability to `bge-large-zh` | `为这个句子生成表示以用于检索相关文章：`  |
|  [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) |   Chinese | [Inference](#usage-for-embedding-model) [Fine-tune](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) | a small-scale model but with competitive performance | `为这个句子生成表示以用于检索相关文章：`  |


[1\]: If you need to search the relevant passages to a query, we suggest to add the instruction to the query; in other cases, no instruction is needed, just use the original query directly. In all cases, **no instruction** needs to be added to passages.

[2\]: Different from embedding model, reranker uses question and document as input and directly output similarity instead of embedding. To balance the accuracy and time cost, cross-encoder is widely used to re-rank top-k documents retrieved by other simple models. 
For examples, use bge embedding model to retrieve top 100 relevant documents, and then use bge reranker to re-rank the top 100 document to get the final top-3 results.

All models have been uploaded to Huggingface Hub, and you can see them at https://huggingface.co/BAAI. 
If you cannot open the Huggingface Hub, you also can download the models at https://model.baai.ac.cn/models .


## Frequently asked questions

<details>
  <summary>1. How to fine-tune bge embedding model?</summary>

  <!-- ### How to fine-tune bge embedding model? -->
Following this [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune) to prepare data and fine-tune your model. 
Some suggestions:
- Mine hard negatives following this [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune#hard-negatives), which can improve the retrieval performance.
- If you pre-train bge on your data, the pre-trained model cannot be directly used to calculate similarity, and it must be fine-tuned with contrastive learning before computing similarity.
- If the accuracy of the fine-tuned model is still not high, it is recommended to use/fine-tune the cross-encoder model (bge-reranker) to re-rank top-k results. Hard negatives also are needed to fine-tune reranker.

  
</details>

<details>
  <summary>2. The similarity score between two dissimilar sentences is higher than 0.5</summary>

  <!-- ### The similarity score between two dissimilar sentences is higher than 0.5 -->
**Suggest to use bge v1.5, which alleviates the issue of the similarity distribution.** 

Since we finetune the models by contrastive learning with a temperature of 0.01, 
the similarity distribution of the current BGE model is about in the interval \[0.6, 1\].
So a similarity score greater than 0.5 does not indicate that the two sentences are similar.

For downstream tasks, such as passage retrieval or semantic similarity, 
**what matters is the relative order of the scores, not the absolute value.**
If you need to filter similar sentences based on a similarity threshold, 
please select an appropriate similarity threshold based on the similarity distribution on your data (such as 0.8, 0.85, or even 0.9).

</details>

<details>
  <summary>3. When does the query instruction need to be used</summary>

  <!-- ### When does the query instruction need to be used -->

For the `bge-*-v1.5`, we improve its retrieval ability when not using instruction. 
No instruction only has a slight degradation in retrieval performance compared with using instruction. 
So you can generate embedding without instruction in all cases for convenience.
 
For a retrieval task that uses short queries to find long related documents, 
it is recommended to add instructions for these short queries.
**The best method to decide whether to add instructions for queries is choosing the setting that achieves better performance on your task.**
In all cases, the documents/passages do not need to add the instruction. 

</details>


## Usage 

### Usage for Embedding Model

Here are some examples for using `bge` models with 
[FlagEmbedding](#using-flagembedding), [Sentence-Transformers](#using-sentence-transformers), [Langchain](#using-langchain), or [Huggingface Transformers](#using-huggingface-transformers).

#### Using FlagEmbedding
```
pip install -U FlagEmbedding
```
If it doesn't work for you, you can see [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md) for more methods to install FlagEmbedding.

```python
from FlagEmbedding import FlagModel
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode() or encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T
```
For the value of the argument `query_instruction_for_retrieval`, see [Model List](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list). 

By default, FlagModel will use all available GPUs when encoding. Please set `os.environ["CUDA_VISIBLE_DEVICES"]` to select specific GPUs.
You also can set `os.environ["CUDA_VISIBLE_DEVICES"]=""` to make all GPUs unavailable.


#### Using Sentence-Transformers

You can also use the `bge` models with [sentence-transformers](https://www.SBERT.net):

```
pip install -U sentence-transformers
```
```python
from sentence_transformers import SentenceTransformer
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```
For s2p(short query to long passage) retrieval task, 
each short query should start with an instruction (instructions see [Model List](https://github.com/FlagOpen/FlagEmbedding/tree/master#model-list)). 
But the instruction is not needed for passages.
```python
from sentence_transformers import SentenceTransformer
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
instruction = "为这个句子生成表示以用于检索相关文章："

model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
q_embeddings = model.encode([instruction+q for q in queries], normalize_embeddings=True)
p_embeddings = model.encode(passages, normalize_embeddings=True)
scores = q_embeddings @ p_embeddings.T
```

#### Using Langchain 

You can use `bge` in langchain like this:
```python
from langchain.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
model.query_instruction = "为这个句子生成表示以用于检索相关文章："
```


#### Using HuggingFace Transformers

With the transformers package, you can use the model like this: First, you pass your input through the transformer model, then you select the last hidden state of the first token (i.e., [CLS]) as the sentence embedding.

```python
from transformers import AutoTokenizer, AutoModel
import torch
# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
model.eval()

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]
# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:", sentence_embeddings)
```

### Usage for Reranker

Different from embedding model, reranker uses question and document as input and directly output similarity instead of embedding. 
You can get a relevance score by inputting query and passage to the reranker. 
The reranker is optimized based cross-entropy loss, so the relevance score is not bounded to a specific range.


#### Using FlagEmbedding
```
pip install -U FlagEmbedding
```

Get relevance scores (higher scores indicate more relevance):
```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'])
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)
```


#### Using Huggingface transformers

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
```

## Evaluation  

`baai-general-embedding` models achieve **state-of-the-art performance on both MTEB and C-MTEB leaderboard!**
For more details and evaluation tools see our [scripts](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md). 

- **MTEB**:   

| Model Name |  Dimension | Sequence Length | Average (56) | Retrieval (15) |Clustering (11) | Pair Classification (3) | Reranking (4) |  STS (10) | Summarization (1) | Classification (12) |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | 1024 | 512 |  **64.23** | **54.29** |  46.08 | 87.12 | 60.03 | 83.11 | 31.61 | 75.97 |  
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) |  768 | 512 | 63.55 | 53.25 |   45.77 | 86.55 | 58.86 | 82.4 | 31.07 | 75.53 |  
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) |  384 | 512 | 62.17 |51.68 | 43.82 |  84.92 | 58.36 | 81.59 | 30.12 | 74.14 |  
| [bge-large-en](https://huggingface.co/BAAI/bge-large-en) |  1024 | 512 | 63.98 |  53.9 | 46.98 | 85.8 | 59.48 | 81.56 | 32.06 | 76.21 | 
| [bge-base-en](https://huggingface.co/BAAI/bge-base-en) |  768 | 512 |  63.36 | 53.0 | 46.32 | 85.86 | 58.7 | 81.84 | 29.27 | 75.27 | 
| [gte-large](https://huggingface.co/thenlper/gte-large) |  1024 | 512 | 63.13 | 52.22 | 46.84 | 85.00 | 59.13 | 83.35 | 31.66 | 73.33 |
| [gte-base](https://huggingface.co/thenlper/gte-base) 	|  768 | 512 | 62.39 | 51.14 | 46.2 | 84.57 | 58.61 | 82.3 | 31.17 | 73.01 |
| [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) |  1024| 512 | 62.25 | 50.56 | 44.49 | 86.03 | 56.61 | 82.05 | 30.19 | 75.24 |
| [bge-small-en](https://huggingface.co/BAAI/bge-small-en) |  384 | 512 | 62.11 |  51.82 | 44.31 | 83.78 | 57.97 | 80.72 | 30.53 | 74.37 |  
| [instructor-xl](https://huggingface.co/hkunlp/instructor-xl) |  768 | 512 | 61.79 | 49.26 | 44.74 | 86.62 | 57.29 | 83.06 | 32.32 | 61.79 |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) |  768 | 512 | 61.5 | 50.29 | 43.80 | 85.73 | 55.91 | 81.05 | 30.28 | 73.84 |
| [gte-small](https://huggingface.co/thenlper/gte-small) |  384 | 512 | 61.36 | 49.46 | 44.89 | 83.54 | 57.7 | 82.07 | 30.42 | 72.31 |
| [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) | 1536 | 8192 | 60.99 | 49.25 | 45.9 | 84.89 | 56.32 | 80.97 | 30.8 | 70.93 |
| [e5-small-v2](https://huggingface.co/intfloat/e5-base-v2) | 384 | 512 | 59.93 | 49.04 | 39.92 | 84.67 | 54.32 | 80.39 | 31.16 | 72.94 |
| [sentence-t5-xxl](https://huggingface.co/sentence-transformers/sentence-t5-xxl) |  768 | 512 | 59.51 | 42.24 | 43.72 | 85.06 | 56.42 | 82.63 | 30.08 | 73.42 |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) 	|  768 | 514 	| 57.78 | 43.81 | 43.69 | 83.04 | 59.36 | 80.28 | 27.49 | 65.07 |
| [sgpt-bloom-7b1-msmarco](https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco) 	|  4096 | 2048 | 57.59 | 48.22 | 38.93 | 81.9 | 55.65 | 77.74 | 33.6 | 66.19 |



- **C-MTEB**:  
We create the benchmark C-MTEB for Chinese text embedding which consists of 31 datasets from 6 tasks. 
Please refer to [C_MTEB](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md) for a detailed introduction.
 
| Model | Embedding dimension | Avg | Retrieval | STS | PairClassification | Classification | Reranking | Clustering |
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| [**BAAI/bge-large-zh-v1.5**](https://huggingface.co/BAAI/bge-large-zh-v1.5) | 1024 |  **64.53** | 70.46 | 56.25 | 81.6 | 69.13 | 65.84 | 48.99 |  
| [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) | 768 |  63.13 | 69.49 | 53.72 | 79.75 | 68.07 | 65.39 | 47.53 |  
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | 512 | 57.82 | 61.77 | 49.11 | 70.41 | 63.96 | 60.92 | 44.18 |   
| [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) | 1024 | 64.20 | 71.53 | 54.98 | 78.94 | 68.32 | 65.11 | 48.39 |
| [bge-large-zh-noinstruct](https://huggingface.co/BAAI/bge-large-zh-noinstruct) | 1024 | 63.53 | 70.55 | 53 | 76.77 | 68.58 | 64.91 | 50.01 |
| [BAAI/bge-base-zh](https://huggingface.co/BAAI/bge-base-zh) | 768 | 62.96 | 69.53 | 54.12 | 77.5 | 67.07 | 64.91 | 47.63 |
| [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | 1024 | 58.79 | 63.66 | 48.44 | 69.89 | 67.34 | 56.00 | 48.23 |
| [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) | 512 | 58.27 |  63.07 | 49.45 | 70.35 | 63.64 | 61.48 | 45.09 |
| [m3e-base](https://huggingface.co/moka-ai/m3e-base) | 768 | 57.10 | 56.91 | 50.47 | 63.99 | 67.52 | 59.34 | 47.68 |
| [m3e-large](https://huggingface.co/moka-ai/m3e-large) | 1024 |  57.05 | 54.75 | 50.42 | 64.3 | 68.2 | 59.66 | 48.88 |
| [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 768 | 55.48 | 61.63 | 46.49 | 67.07 | 65.35 | 54.35 | 40.68 |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 384 | 55.38 | 59.95 | 45.27 | 66.45 | 65.85 | 53.86 | 45.26 |
| [text-embedding-ada-002(OpenAI)](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) | 1536 |  53.02 | 52.0 | 43.35 | 69.56 | 64.31 | 54.28 | 45.68 |
| [luotuo](https://huggingface.co/silk-road/luotuo-bert-medium) | 1024 | 49.37 |  44.4 | 42.78 | 66.62 | 61 | 49.25 | 44.39 |
| [text2vec-base](https://huggingface.co/shibing624/text2vec-base-chinese) | 768 |  47.63 | 38.79 | 43.41 | 67.41 | 62.19 | 49.45 | 37.66 |
| [text2vec-large](https://huggingface.co/GanymedeNil/text2vec-large-chinese) | 1024 | 47.36 | 41.94 | 44.97 | 70.86 | 60.66 | 49.16 | 30.02 |


- **Reranking**:
See [C_MTEB](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/) for evaluation script.

| Model | T2Reranking | T2RerankingZh2En\* | T2RerankingEn2Zh\* | MMarcoReranking | CMedQAv1 | CMedQAv2 | Avg |  
|:-------------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|  
| text2vec-base-multilingual | 64.66 | 62.94 | 62.51 | 14.37 | 48.46 | 48.6 | 50.26 |  
| multilingual-e5-small | 65.62 | 60.94 | 56.41 | 29.91 | 67.26 | 66.54 | 57.78 |  
| multilingual-e5-large | 64.55 | 61.61 | 54.28 | 28.6 | 67.42 | 67.92 | 57.4 |  
| multilingual-e5-base | 64.21 | 62.13 | 54.68 | 29.5 | 66.23 | 66.98 | 57.29 |  
| m3e-base | 66.03 | 62.74 | 56.07 | 17.51 | 77.05 | 76.76 | 59.36 |  
| m3e-large | 66.13 | 62.72 | 56.1 | 16.46 | 77.76 | 78.27 | 59.57 |  
| bge-base-zh-v1.5 | 66.49 | 63.25 | 57.02 | 29.74 | 80.47 | 84.88 | 63.64 |  
| bge-large-zh-v1.5 | 65.74 | 63.39 | 57.03 | 28.74 | 83.45 | 85.44 | 63.97 |  
| [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | 67.28 | 63.95 | 60.45 | 35.46 | 81.26 | 84.1 | 65.42 |  
| [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | 67.6 | 64.03 | 61.44 | 37.16 | 82.15 | 84.18 | 66.09 |  

\* : T2RerankingZh2En and T2RerankingEn2Zh are cross-language retrieval tasks

## Train

### BAAI Embedding 

We pre-train the models using [retromae](https://github.com/staoxiao/RetroMAE) and train them on large-scale pairs data using contrastive learning. 
**You can fine-tune the embedding model on your data following our [examples](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/finetune).**
We also provide a [pre-train example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/pretrain).
Note that the goal of pre-training is to reconstruct the text, and the pre-trained model cannot be used for similarity calculation directly, it needs to be fine-tuned.
More training details for bge see [baai_general_embedding](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/baai_general_embedding/README.md).



### BGE Reranker

Cross-encoder will perform full-attention over the input pair, 
which is more accurate than embedding model (i.e., bi-encoder) but more time-consuming than embedding model.
Therefore, it can be used to re-rank the top-k documents returned by embedding model.
We train the cross-encoder on a multilingual pair data, 
The data format is the same as embedding model, so you can fine-tune it easily following our [example](https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/reranker). 
More details please refer to [./FlagEmbedding/reranker/README.md](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)


## Contact
If you have any question or suggestion related to this project, feel free to open an issue or pull request.
You also can email Shitao Xiao(stxiao@baai.ac.cn) and Zheng Liu(liuzheng@baai.ac.cn). 


## Citation

If you find this repository useful, please consider giving a star :star: and citation

```
@misc{bge_embedding,
      title={C-Pack: Packaged Resources To Advance General Chinese Embedding}, 
      author={Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff},
      year={2023},
      eprint={2309.07597},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
FlagEmbedding is licensed under the [MIT License](https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE). The released models can be used for commercial purposes free of charge.

