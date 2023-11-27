---
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- transformers
- mteb
model-index:
- name: bge-large-en-v1.5
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
      value: 75.8507462686567
    - type: ap
      value: 38.566457320228245
    - type: f1
      value: 69.69386648043475
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
      value: 92.416675
    - type: ap
      value: 89.1928861155922
    - type: f1
      value: 92.39477019574215
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
      value: 48.175999999999995
    - type: f1
      value: 47.80712792870253
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
      value: 40.184999999999995
    - type: map_at_10
      value: 55.654
    - type: map_at_100
      value: 56.25
    - type: map_at_1000
      value: 56.255
    - type: map_at_3
      value: 51.742999999999995
    - type: map_at_5
      value: 54.129000000000005
    - type: mrr_at_1
      value: 40.967
    - type: mrr_at_10
      value: 55.96
    - type: mrr_at_100
      value: 56.54900000000001
    - type: mrr_at_1000
      value: 56.554
    - type: mrr_at_3
      value: 51.980000000000004
    - type: mrr_at_5
      value: 54.44
    - type: ndcg_at_1
      value: 40.184999999999995
    - type: ndcg_at_10
      value: 63.542
    - type: ndcg_at_100
      value: 65.96499999999999
    - type: ndcg_at_1000
      value: 66.08699999999999
    - type: ndcg_at_3
      value: 55.582
    - type: ndcg_at_5
      value: 59.855000000000004
    - type: precision_at_1
      value: 40.184999999999995
    - type: precision_at_10
      value: 8.841000000000001
    - type: precision_at_100
      value: 0.987
    - type: precision_at_1000
      value: 0.1
    - type: precision_at_3
      value: 22.238
    - type: precision_at_5
      value: 15.405
    - type: recall_at_1
      value: 40.184999999999995
    - type: recall_at_10
      value: 88.407
    - type: recall_at_100
      value: 98.72
    - type: recall_at_1000
      value: 99.644
    - type: recall_at_3
      value: 66.714
    - type: recall_at_5
      value: 77.027
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
      value: 48.567077926750066
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
      value: 43.19453389182364
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
      value: 64.46555939623092
    - type: mrr
      value: 77.82361605768807
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
      value: 84.9554128814735
    - type: cos_sim_spearman
      value: 84.65373612172036
    - type: euclidean_pearson
      value: 83.2905059954138
    - type: euclidean_spearman
      value: 84.52240782811128
    - type: manhattan_pearson
      value: 82.99533802997436
    - type: manhattan_spearman
      value: 84.20673798475734
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
      value: 87.78896103896103
    - type: f1
      value: 87.77189310964883
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
      value: 39.714538337650495
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
      value: 36.90108349284447
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
      value: 32.795
    - type: map_at_10
      value: 43.669000000000004
    - type: map_at_100
      value: 45.151
    - type: map_at_1000
      value: 45.278
    - type: map_at_3
      value: 40.006
    - type: map_at_5
      value: 42.059999999999995
    - type: mrr_at_1
      value: 39.771
    - type: mrr_at_10
      value: 49.826
    - type: mrr_at_100
      value: 50.504000000000005
    - type: mrr_at_1000
      value: 50.549
    - type: mrr_at_3
      value: 47.115
    - type: mrr_at_5
      value: 48.832
    - type: ndcg_at_1
      value: 39.771
    - type: ndcg_at_10
      value: 50.217999999999996
    - type: ndcg_at_100
      value: 55.454
    - type: ndcg_at_1000
      value: 57.37
    - type: ndcg_at_3
      value: 44.885000000000005
    - type: ndcg_at_5
      value: 47.419
    - type: precision_at_1
      value: 39.771
    - type: precision_at_10
      value: 9.642000000000001
    - type: precision_at_100
      value: 1.538
    - type: precision_at_1000
      value: 0.198
    - type: precision_at_3
      value: 21.268
    - type: precision_at_5
      value: 15.536
    - type: recall_at_1
      value: 32.795
    - type: recall_at_10
      value: 62.580999999999996
    - type: recall_at_100
      value: 84.438
    - type: recall_at_1000
      value: 96.492
    - type: recall_at_3
      value: 47.071000000000005
    - type: recall_at_5
      value: 54.079
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
      value: 43.334
    - type: map_at_100
      value: 44.566
    - type: map_at_1000
      value: 44.702999999999996
    - type: map_at_3
      value: 40.343
    - type: map_at_5
      value: 41.983
    - type: mrr_at_1
      value: 40.764
    - type: mrr_at_10
      value: 49.382
    - type: mrr_at_100
      value: 49.988
    - type: mrr_at_1000
      value: 50.03300000000001
    - type: mrr_at_3
      value: 47.293
    - type: mrr_at_5
      value: 48.51
    - type: ndcg_at_1
      value: 40.764
    - type: ndcg_at_10
      value: 49.039
    - type: ndcg_at_100
      value: 53.259
    - type: ndcg_at_1000
      value: 55.253
    - type: ndcg_at_3
      value: 45.091
    - type: ndcg_at_5
      value: 46.839999999999996
    - type: precision_at_1
      value: 40.764
    - type: precision_at_10
      value: 9.191
    - type: precision_at_100
      value: 1.476
    - type: precision_at_1000
      value: 0.19499999999999998
    - type: precision_at_3
      value: 21.72
    - type: precision_at_5
      value: 15.299
    - type: recall_at_1
      value: 32.671
    - type: recall_at_10
      value: 58.816
    - type: recall_at_100
      value: 76.654
    - type: recall_at_1000
      value: 89.05999999999999
    - type: recall_at_3
      value: 46.743
    - type: recall_at_5
      value: 51.783
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
      value: 40.328
    - type: map_at_10
      value: 53.32599999999999
    - type: map_at_100
      value: 54.37499999999999
    - type: map_at_1000
      value: 54.429
    - type: map_at_3
      value: 49.902
    - type: map_at_5
      value: 52.002
    - type: mrr_at_1
      value: 46.332
    - type: mrr_at_10
      value: 56.858
    - type: mrr_at_100
      value: 57.522
    - type: mrr_at_1000
      value: 57.54899999999999
    - type: mrr_at_3
      value: 54.472
    - type: mrr_at_5
      value: 55.996
    - type: ndcg_at_1
      value: 46.332
    - type: ndcg_at_10
      value: 59.313
    - type: ndcg_at_100
      value: 63.266999999999996
    - type: ndcg_at_1000
      value: 64.36
    - type: ndcg_at_3
      value: 53.815000000000005
    - type: ndcg_at_5
      value: 56.814
    - type: precision_at_1
      value: 46.332
    - type: precision_at_10
      value: 9.53
    - type: precision_at_100
      value: 1.238
    - type: precision_at_1000
      value: 0.13699999999999998
    - type: precision_at_3
      value: 24.054000000000002
    - type: precision_at_5
      value: 16.589000000000002
    - type: recall_at_1
      value: 40.328
    - type: recall_at_10
      value: 73.421
    - type: recall_at_100
      value: 90.059
    - type: recall_at_1000
      value: 97.81
    - type: recall_at_3
      value: 59.009
    - type: recall_at_5
      value: 66.352
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
      value: 27.424
    - type: map_at_10
      value: 36.332
    - type: map_at_100
      value: 37.347
    - type: map_at_1000
      value: 37.422
    - type: map_at_3
      value: 33.743
    - type: map_at_5
      value: 35.176
    - type: mrr_at_1
      value: 29.153000000000002
    - type: mrr_at_10
      value: 38.233
    - type: mrr_at_100
      value: 39.109
    - type: mrr_at_1000
      value: 39.164
    - type: mrr_at_3
      value: 35.876000000000005
    - type: mrr_at_5
      value: 37.169000000000004
    - type: ndcg_at_1
      value: 29.153000000000002
    - type: ndcg_at_10
      value: 41.439
    - type: ndcg_at_100
      value: 46.42
    - type: ndcg_at_1000
      value: 48.242000000000004
    - type: ndcg_at_3
      value: 36.362
    - type: ndcg_at_5
      value: 38.743
    - type: precision_at_1
      value: 29.153000000000002
    - type: precision_at_10
      value: 6.315999999999999
    - type: precision_at_100
      value: 0.927
    - type: precision_at_1000
      value: 0.11199999999999999
    - type: precision_at_3
      value: 15.443000000000001
    - type: precision_at_5
      value: 10.644
    - type: recall_at_1
      value: 27.424
    - type: recall_at_10
      value: 55.364000000000004
    - type: recall_at_100
      value: 78.211
    - type: recall_at_1000
      value: 91.74600000000001
    - type: recall_at_3
      value: 41.379
    - type: recall_at_5
      value: 47.14
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
      value: 19.601
    - type: map_at_10
      value: 27.826
    - type: map_at_100
      value: 29.017
    - type: map_at_1000
      value: 29.137
    - type: map_at_3
      value: 25.125999999999998
    - type: map_at_5
      value: 26.765
    - type: mrr_at_1
      value: 24.005000000000003
    - type: mrr_at_10
      value: 32.716
    - type: mrr_at_100
      value: 33.631
    - type: mrr_at_1000
      value: 33.694
    - type: mrr_at_3
      value: 29.934
    - type: mrr_at_5
      value: 31.630999999999997
    - type: ndcg_at_1
      value: 24.005000000000003
    - type: ndcg_at_10
      value: 33.158
    - type: ndcg_at_100
      value: 38.739000000000004
    - type: ndcg_at_1000
      value: 41.495
    - type: ndcg_at_3
      value: 28.185
    - type: ndcg_at_5
      value: 30.796
    - type: precision_at_1
      value: 24.005000000000003
    - type: precision_at_10
      value: 5.908
    - type: precision_at_100
      value: 1.005
    - type: precision_at_1000
      value: 0.13899999999999998
    - type: precision_at_3
      value: 13.391
    - type: precision_at_5
      value: 9.876
    - type: recall_at_1
      value: 19.601
    - type: recall_at_10
      value: 44.746
    - type: recall_at_100
      value: 68.82300000000001
    - type: recall_at_1000
      value: 88.215
    - type: recall_at_3
      value: 31.239
    - type: recall_at_5
      value: 37.695
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
      value: 30.130000000000003
    - type: map_at_10
      value: 40.96
    - type: map_at_100
      value: 42.282
    - type: map_at_1000
      value: 42.392
    - type: map_at_3
      value: 37.889
    - type: map_at_5
      value: 39.661
    - type: mrr_at_1
      value: 36.958999999999996
    - type: mrr_at_10
      value: 46.835
    - type: mrr_at_100
      value: 47.644
    - type: mrr_at_1000
      value: 47.688
    - type: mrr_at_3
      value: 44.562000000000005
    - type: mrr_at_5
      value: 45.938
    - type: ndcg_at_1
      value: 36.958999999999996
    - type: ndcg_at_10
      value: 47.06
    - type: ndcg_at_100
      value: 52.345
    - type: ndcg_at_1000
      value: 54.35
    - type: ndcg_at_3
      value: 42.301
    - type: ndcg_at_5
      value: 44.635999999999996
    - type: precision_at_1
      value: 36.958999999999996
    - type: precision_at_10
      value: 8.479000000000001
    - type: precision_at_100
      value: 1.284
    - type: precision_at_1000
      value: 0.163
    - type: precision_at_3
      value: 20.244
    - type: precision_at_5
      value: 14.224999999999998
    - type: recall_at_1
      value: 30.130000000000003
    - type: recall_at_10
      value: 59.27
    - type: recall_at_100
      value: 81.195
    - type: recall_at_1000
      value: 94.21199999999999
    - type: recall_at_3
      value: 45.885
    - type: recall_at_5
      value: 52.016
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
      value: 26.169999999999998
    - type: map_at_10
      value: 36.451
    - type: map_at_100
      value: 37.791000000000004
    - type: map_at_1000
      value: 37.897
    - type: map_at_3
      value: 33.109
    - type: map_at_5
      value: 34.937000000000005
    - type: mrr_at_1
      value: 32.877
    - type: mrr_at_10
      value: 42.368
    - type: mrr_at_100
      value: 43.201
    - type: mrr_at_1000
      value: 43.259
    - type: mrr_at_3
      value: 39.763999999999996
    - type: mrr_at_5
      value: 41.260000000000005
    - type: ndcg_at_1
      value: 32.877
    - type: ndcg_at_10
      value: 42.659000000000006
    - type: ndcg_at_100
      value: 48.161
    - type: ndcg_at_1000
      value: 50.345
    - type: ndcg_at_3
      value: 37.302
    - type: ndcg_at_5
      value: 39.722
    - type: precision_at_1
      value: 32.877
    - type: precision_at_10
      value: 7.9
    - type: precision_at_100
      value: 1.236
    - type: precision_at_1000
      value: 0.158
    - type: precision_at_3
      value: 17.846
    - type: precision_at_5
      value: 12.9
    - type: recall_at_1
      value: 26.169999999999998
    - type: recall_at_10
      value: 55.35
    - type: recall_at_100
      value: 78.755
    - type: recall_at_1000
      value: 93.518
    - type: recall_at_3
      value: 40.176
    - type: recall_at_5
      value: 46.589000000000006
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
      value: 27.15516666666667
    - type: map_at_10
      value: 36.65741666666667
    - type: map_at_100
      value: 37.84991666666666
    - type: map_at_1000
      value: 37.96316666666667
    - type: map_at_3
      value: 33.74974999999999
    - type: map_at_5
      value: 35.3765
    - type: mrr_at_1
      value: 32.08233333333334
    - type: mrr_at_10
      value: 41.033833333333334
    - type: mrr_at_100
      value: 41.84524999999999
    - type: mrr_at_1000
      value: 41.89983333333333
    - type: mrr_at_3
      value: 38.62008333333333
    - type: mrr_at_5
      value: 40.03441666666666
    - type: ndcg_at_1
      value: 32.08233333333334
    - type: ndcg_at_10
      value: 42.229
    - type: ndcg_at_100
      value: 47.26716666666667
    - type: ndcg_at_1000
      value: 49.43466666666667
    - type: ndcg_at_3
      value: 37.36408333333333
    - type: ndcg_at_5
      value: 39.6715
    - type: precision_at_1
      value: 32.08233333333334
    - type: precision_at_10
      value: 7.382583333333334
    - type: precision_at_100
      value: 1.16625
    - type: precision_at_1000
      value: 0.15408333333333332
    - type: precision_at_3
      value: 17.218
    - type: precision_at_5
      value: 12.21875
    - type: recall_at_1
      value: 27.15516666666667
    - type: recall_at_10
      value: 54.36683333333333
    - type: recall_at_100
      value: 76.37183333333333
    - type: recall_at_1000
      value: 91.26183333333333
    - type: recall_at_3
      value: 40.769916666666674
    - type: recall_at_5
      value: 46.702333333333335
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
      value: 25.749
    - type: map_at_10
      value: 33.001999999999995
    - type: map_at_100
      value: 33.891
    - type: map_at_1000
      value: 33.993
    - type: map_at_3
      value: 30.703999999999997
    - type: map_at_5
      value: 31.959
    - type: mrr_at_1
      value: 28.834
    - type: mrr_at_10
      value: 35.955
    - type: mrr_at_100
      value: 36.709
    - type: mrr_at_1000
      value: 36.779
    - type: mrr_at_3
      value: 33.947
    - type: mrr_at_5
      value: 35.089
    - type: ndcg_at_1
      value: 28.834
    - type: ndcg_at_10
      value: 37.329
    - type: ndcg_at_100
      value: 41.79
    - type: ndcg_at_1000
      value: 44.169000000000004
    - type: ndcg_at_3
      value: 33.184999999999995
    - type: ndcg_at_5
      value: 35.107
    - type: precision_at_1
      value: 28.834
    - type: precision_at_10
      value: 5.7669999999999995
    - type: precision_at_100
      value: 0.876
    - type: precision_at_1000
      value: 0.11399999999999999
    - type: precision_at_3
      value: 14.213000000000001
    - type: precision_at_5
      value: 9.754999999999999
    - type: recall_at_1
      value: 25.749
    - type: recall_at_10
      value: 47.791
    - type: recall_at_100
      value: 68.255
    - type: recall_at_1000
      value: 85.749
    - type: recall_at_3
      value: 36.199
    - type: recall_at_5
      value: 41.071999999999996
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
      value: 17.777
    - type: map_at_10
      value: 25.201
    - type: map_at_100
      value: 26.423999999999996
    - type: map_at_1000
      value: 26.544
    - type: map_at_3
      value: 22.869
    - type: map_at_5
      value: 24.023
    - type: mrr_at_1
      value: 21.473
    - type: mrr_at_10
      value: 29.12
    - type: mrr_at_100
      value: 30.144
    - type: mrr_at_1000
      value: 30.215999999999998
    - type: mrr_at_3
      value: 26.933
    - type: mrr_at_5
      value: 28.051
    - type: ndcg_at_1
      value: 21.473
    - type: ndcg_at_10
      value: 30.003
    - type: ndcg_at_100
      value: 35.766
    - type: ndcg_at_1000
      value: 38.501000000000005
    - type: ndcg_at_3
      value: 25.773000000000003
    - type: ndcg_at_5
      value: 27.462999999999997
    - type: precision_at_1
      value: 21.473
    - type: precision_at_10
      value: 5.482
    - type: precision_at_100
      value: 0.975
    - type: precision_at_1000
      value: 0.13799999999999998
    - type: precision_at_3
      value: 12.205
    - type: precision_at_5
      value: 8.692
    - type: recall_at_1
      value: 17.777
    - type: recall_at_10
      value: 40.582
    - type: recall_at_100
      value: 66.305
    - type: recall_at_1000
      value: 85.636
    - type: recall_at_3
      value: 28.687
    - type: recall_at_5
      value: 33.089
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
      value: 26.677
    - type: map_at_10
      value: 36.309000000000005
    - type: map_at_100
      value: 37.403999999999996
    - type: map_at_1000
      value: 37.496
    - type: map_at_3
      value: 33.382
    - type: map_at_5
      value: 34.98
    - type: mrr_at_1
      value: 31.343
    - type: mrr_at_10
      value: 40.549
    - type: mrr_at_100
      value: 41.342
    - type: mrr_at_1000
      value: 41.397
    - type: mrr_at_3
      value: 38.029
    - type: mrr_at_5
      value: 39.451
    - type: ndcg_at_1
      value: 31.343
    - type: ndcg_at_10
      value: 42.1
    - type: ndcg_at_100
      value: 47.089999999999996
    - type: ndcg_at_1000
      value: 49.222
    - type: ndcg_at_3
      value: 36.836999999999996
    - type: ndcg_at_5
      value: 39.21
    - type: precision_at_1
      value: 31.343
    - type: precision_at_10
      value: 7.164
    - type: precision_at_100
      value: 1.0959999999999999
    - type: precision_at_1000
      value: 0.13899999999999998
    - type: precision_at_3
      value: 16.915
    - type: precision_at_5
      value: 11.940000000000001
    - type: recall_at_1
      value: 26.677
    - type: recall_at_10
      value: 55.54599999999999
    - type: recall_at_100
      value: 77.094
    - type: recall_at_1000
      value: 92.01
    - type: recall_at_3
      value: 41.191
    - type: recall_at_5
      value: 47.006
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
      value: 24.501
    - type: map_at_10
      value: 33.102
    - type: map_at_100
      value: 34.676
    - type: map_at_1000
      value: 34.888000000000005
    - type: map_at_3
      value: 29.944
    - type: map_at_5
      value: 31.613999999999997
    - type: mrr_at_1
      value: 29.447000000000003
    - type: mrr_at_10
      value: 37.996
    - type: mrr_at_100
      value: 38.946
    - type: mrr_at_1000
      value: 38.995000000000005
    - type: mrr_at_3
      value: 35.079
    - type: mrr_at_5
      value: 36.69
    - type: ndcg_at_1
      value: 29.447000000000003
    - type: ndcg_at_10
      value: 39.232
    - type: ndcg_at_100
      value: 45.247
    - type: ndcg_at_1000
      value: 47.613
    - type: ndcg_at_3
      value: 33.922999999999995
    - type: ndcg_at_5
      value: 36.284
    - type: precision_at_1
      value: 29.447000000000003
    - type: precision_at_10
      value: 7.648000000000001
    - type: precision_at_100
      value: 1.516
    - type: precision_at_1000
      value: 0.23900000000000002
    - type: precision_at_3
      value: 16.008
    - type: precision_at_5
      value: 11.779
    - type: recall_at_1
      value: 24.501
    - type: recall_at_10
      value: 51.18899999999999
    - type: recall_at_100
      value: 78.437
    - type: recall_at_1000
      value: 92.842
    - type: recall_at_3
      value: 35.808
    - type: recall_at_5
      value: 42.197
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
      value: 22.039
    - type: map_at_10
      value: 30.377
    - type: map_at_100
      value: 31.275
    - type: map_at_1000
      value: 31.379
    - type: map_at_3
      value: 27.98
    - type: map_at_5
      value: 29.358
    - type: mrr_at_1
      value: 24.03
    - type: mrr_at_10
      value: 32.568000000000005
    - type: mrr_at_100
      value: 33.403
    - type: mrr_at_1000
      value: 33.475
    - type: mrr_at_3
      value: 30.436999999999998
    - type: mrr_at_5
      value: 31.796000000000003
    - type: ndcg_at_1
      value: 24.03
    - type: ndcg_at_10
      value: 35.198
    - type: ndcg_at_100
      value: 39.668
    - type: ndcg_at_1000
      value: 42.296
    - type: ndcg_at_3
      value: 30.709999999999997
    - type: ndcg_at_5
      value: 33.024
    - type: precision_at_1
      value: 24.03
    - type: precision_at_10
      value: 5.564
    - type: precision_at_100
      value: 0.828
    - type: precision_at_1000
      value: 0.117
    - type: precision_at_3
      value: 13.309000000000001
    - type: precision_at_5
      value: 9.39
    - type: recall_at_1
      value: 22.039
    - type: recall_at_10
      value: 47.746
    - type: recall_at_100
      value: 68.23599999999999
    - type: recall_at_1000
      value: 87.852
    - type: recall_at_3
      value: 35.852000000000004
    - type: recall_at_5
      value: 41.410000000000004
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
      value: 15.692999999999998
    - type: map_at_10
      value: 26.903
    - type: map_at_100
      value: 28.987000000000002
    - type: map_at_1000
      value: 29.176999999999996
    - type: map_at_3
      value: 22.137
    - type: map_at_5
      value: 24.758
    - type: mrr_at_1
      value: 35.57
    - type: mrr_at_10
      value: 47.821999999999996
    - type: mrr_at_100
      value: 48.608000000000004
    - type: mrr_at_1000
      value: 48.638999999999996
    - type: mrr_at_3
      value: 44.452000000000005
    - type: mrr_at_5
      value: 46.546
    - type: ndcg_at_1
      value: 35.57
    - type: ndcg_at_10
      value: 36.567
    - type: ndcg_at_100
      value: 44.085
    - type: ndcg_at_1000
      value: 47.24
    - type: ndcg_at_3
      value: 29.964000000000002
    - type: ndcg_at_5
      value: 32.511
    - type: precision_at_1
      value: 35.57
    - type: precision_at_10
      value: 11.485
    - type: precision_at_100
      value: 1.9619999999999997
    - type: precision_at_1000
      value: 0.256
    - type: precision_at_3
      value: 22.237000000000002
    - type: precision_at_5
      value: 17.471999999999998
    - type: recall_at_1
      value: 15.692999999999998
    - type: recall_at_10
      value: 43.056
    - type: recall_at_100
      value: 68.628
    - type: recall_at_1000
      value: 86.075
    - type: recall_at_3
      value: 26.918999999999997
    - type: recall_at_5
      value: 34.14
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
      value: 9.53
    - type: map_at_10
      value: 20.951
    - type: map_at_100
      value: 30.136000000000003
    - type: map_at_1000
      value: 31.801000000000002
    - type: map_at_3
      value: 15.021
    - type: map_at_5
      value: 17.471999999999998
    - type: mrr_at_1
      value: 71.0
    - type: mrr_at_10
      value: 79.176
    - type: mrr_at_100
      value: 79.418
    - type: mrr_at_1000
      value: 79.426
    - type: mrr_at_3
      value: 78.125
    - type: mrr_at_5
      value: 78.61200000000001
    - type: ndcg_at_1
      value: 58.5
    - type: ndcg_at_10
      value: 44.106
    - type: ndcg_at_100
      value: 49.268
    - type: ndcg_at_1000
      value: 56.711999999999996
    - type: ndcg_at_3
      value: 48.934
    - type: ndcg_at_5
      value: 45.826
    - type: precision_at_1
      value: 71.0
    - type: precision_at_10
      value: 35.0
    - type: precision_at_100
      value: 11.360000000000001
    - type: precision_at_1000
      value: 2.046
    - type: precision_at_3
      value: 52.833
    - type: precision_at_5
      value: 44.15
    - type: recall_at_1
      value: 9.53
    - type: recall_at_10
      value: 26.811
    - type: recall_at_100
      value: 55.916999999999994
    - type: recall_at_1000
      value: 79.973
    - type: recall_at_3
      value: 16.413
    - type: recall_at_5
      value: 19.980999999999998
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
      value: 51.519999999999996
    - type: f1
      value: 46.36601294761231
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
      value: 74.413
    - type: map_at_10
      value: 83.414
    - type: map_at_100
      value: 83.621
    - type: map_at_1000
      value: 83.635
    - type: map_at_3
      value: 82.337
    - type: map_at_5
      value: 83.039
    - type: mrr_at_1
      value: 80.19800000000001
    - type: mrr_at_10
      value: 87.715
    - type: mrr_at_100
      value: 87.778
    - type: mrr_at_1000
      value: 87.779
    - type: mrr_at_3
      value: 87.106
    - type: mrr_at_5
      value: 87.555
    - type: ndcg_at_1
      value: 80.19800000000001
    - type: ndcg_at_10
      value: 87.182
    - type: ndcg_at_100
      value: 87.90299999999999
    - type: ndcg_at_1000
      value: 88.143
    - type: ndcg_at_3
      value: 85.60600000000001
    - type: ndcg_at_5
      value: 86.541
    - type: precision_at_1
      value: 80.19800000000001
    - type: precision_at_10
      value: 10.531
    - type: precision_at_100
      value: 1.113
    - type: precision_at_1000
      value: 0.11499999999999999
    - type: precision_at_3
      value: 32.933
    - type: precision_at_5
      value: 20.429
    - type: recall_at_1
      value: 74.413
    - type: recall_at_10
      value: 94.363
    - type: recall_at_100
      value: 97.165
    - type: recall_at_1000
      value: 98.668
    - type: recall_at_3
      value: 90.108
    - type: recall_at_5
      value: 92.52
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
      value: 22.701
    - type: map_at_10
      value: 37.122
    - type: map_at_100
      value: 39.178000000000004
    - type: map_at_1000
      value: 39.326
    - type: map_at_3
      value: 32.971000000000004
    - type: map_at_5
      value: 35.332
    - type: mrr_at_1
      value: 44.753
    - type: mrr_at_10
      value: 53.452
    - type: mrr_at_100
      value: 54.198
    - type: mrr_at_1000
      value: 54.225
    - type: mrr_at_3
      value: 50.952
    - type: mrr_at_5
      value: 52.464
    - type: ndcg_at_1
      value: 44.753
    - type: ndcg_at_10
      value: 45.021
    - type: ndcg_at_100
      value: 52.028
    - type: ndcg_at_1000
      value: 54.596000000000004
    - type: ndcg_at_3
      value: 41.622
    - type: ndcg_at_5
      value: 42.736000000000004
    - type: precision_at_1
      value: 44.753
    - type: precision_at_10
      value: 12.284
    - type: precision_at_100
      value: 1.955
    - type: precision_at_1000
      value: 0.243
    - type: precision_at_3
      value: 27.828999999999997
    - type: precision_at_5
      value: 20.061999999999998
    - type: recall_at_1
      value: 22.701
    - type: recall_at_10
      value: 51.432
    - type: recall_at_100
      value: 77.009
    - type: recall_at_1000
      value: 92.511
    - type: recall_at_3
      value: 37.919000000000004
    - type: recall_at_5
      value: 44.131
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
      value: 40.189
    - type: map_at_10
      value: 66.24600000000001
    - type: map_at_100
      value: 67.098
    - type: map_at_1000
      value: 67.149
    - type: map_at_3
      value: 62.684
    - type: map_at_5
      value: 64.974
    - type: mrr_at_1
      value: 80.378
    - type: mrr_at_10
      value: 86.127
    - type: mrr_at_100
      value: 86.29299999999999
    - type: mrr_at_1000
      value: 86.297
    - type: mrr_at_3
      value: 85.31400000000001
    - type: mrr_at_5
      value: 85.858
    - type: ndcg_at_1
      value: 80.378
    - type: ndcg_at_10
      value: 74.101
    - type: ndcg_at_100
      value: 76.993
    - type: ndcg_at_1000
      value: 77.948
    - type: ndcg_at_3
      value: 69.232
    - type: ndcg_at_5
      value: 72.04599999999999
    - type: precision_at_1
      value: 80.378
    - type: precision_at_10
      value: 15.595999999999998
    - type: precision_at_100
      value: 1.7840000000000003
    - type: precision_at_1000
      value: 0.191
    - type: precision_at_3
      value: 44.884
    - type: precision_at_5
      value: 29.145
    - type: recall_at_1
      value: 40.189
    - type: recall_at_10
      value: 77.981
    - type: recall_at_100
      value: 89.21
    - type: recall_at_1000
      value: 95.48299999999999
    - type: recall_at_3
      value: 67.326
    - type: recall_at_5
      value: 72.863
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
      value: 92.84599999999999
    - type: ap
      value: 89.4710787567357
    - type: f1
      value: 92.83752676932258
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
      value: 23.132
    - type: map_at_10
      value: 35.543
    - type: map_at_100
      value: 36.702
    - type: map_at_1000
      value: 36.748999999999995
    - type: map_at_3
      value: 31.737
    - type: map_at_5
      value: 33.927
    - type: mrr_at_1
      value: 23.782
    - type: mrr_at_10
      value: 36.204
    - type: mrr_at_100
      value: 37.29
    - type: mrr_at_1000
      value: 37.330999999999996
    - type: mrr_at_3
      value: 32.458999999999996
    - type: mrr_at_5
      value: 34.631
    - type: ndcg_at_1
      value: 23.782
    - type: ndcg_at_10
      value: 42.492999999999995
    - type: ndcg_at_100
      value: 47.985
    - type: ndcg_at_1000
      value: 49.141
    - type: ndcg_at_3
      value: 34.748000000000005
    - type: ndcg_at_5
      value: 38.651
    - type: precision_at_1
      value: 23.782
    - type: precision_at_10
      value: 6.665
    - type: precision_at_100
      value: 0.941
    - type: precision_at_1000
      value: 0.104
    - type: precision_at_3
      value: 14.776
    - type: precision_at_5
      value: 10.84
    - type: recall_at_1
      value: 23.132
    - type: recall_at_10
      value: 63.794
    - type: recall_at_100
      value: 89.027
    - type: recall_at_1000
      value: 97.807
    - type: recall_at_3
      value: 42.765
    - type: recall_at_5
      value: 52.11
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
      value: 94.59188326493388
    - type: f1
      value: 94.3842594786827
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
      value: 79.49384404924761
    - type: f1
      value: 59.7580539534629
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
      value: 77.56220578345663
    - type: f1
      value: 75.27228165561478
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
      value: 80.53463349024884
    - type: f1
      value: 80.4893958236536
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
      value: 32.56100273484962
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
      value: 31.470380028839607
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
      value: 32.06102792457849
    - type: mrr
      value: 33.30709199672238
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
      value: 6.776999999999999
    - type: map_at_10
      value: 14.924000000000001
    - type: map_at_100
      value: 18.955
    - type: map_at_1000
      value: 20.538999999999998
    - type: map_at_3
      value: 10.982
    - type: map_at_5
      value: 12.679000000000002
    - type: mrr_at_1
      value: 47.988
    - type: mrr_at_10
      value: 57.232000000000006
    - type: mrr_at_100
      value: 57.818999999999996
    - type: mrr_at_1000
      value: 57.847
    - type: mrr_at_3
      value: 54.901999999999994
    - type: mrr_at_5
      value: 56.481
    - type: ndcg_at_1
      value: 46.594
    - type: ndcg_at_10
      value: 38.129000000000005
    - type: ndcg_at_100
      value: 35.54
    - type: ndcg_at_1000
      value: 44.172
    - type: ndcg_at_3
      value: 43.025999999999996
    - type: ndcg_at_5
      value: 41.052
    - type: precision_at_1
      value: 47.988
    - type: precision_at_10
      value: 28.111000000000004
    - type: precision_at_100
      value: 8.929
    - type: precision_at_1000
      value: 2.185
    - type: precision_at_3
      value: 40.144000000000005
    - type: precision_at_5
      value: 35.232
    - type: recall_at_1
      value: 6.776999999999999
    - type: recall_at_10
      value: 19.289
    - type: recall_at_100
      value: 36.359
    - type: recall_at_1000
      value: 67.54
    - type: recall_at_3
      value: 11.869
    - type: recall_at_5
      value: 14.999
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
      value: 31.108000000000004
    - type: map_at_10
      value: 47.126000000000005
    - type: map_at_100
      value: 48.171
    - type: map_at_1000
      value: 48.199
    - type: map_at_3
      value: 42.734
    - type: map_at_5
      value: 45.362
    - type: mrr_at_1
      value: 34.936
    - type: mrr_at_10
      value: 49.571
    - type: mrr_at_100
      value: 50.345
    - type: mrr_at_1000
      value: 50.363
    - type: mrr_at_3
      value: 45.959
    - type: mrr_at_5
      value: 48.165
    - type: ndcg_at_1
      value: 34.936
    - type: ndcg_at_10
      value: 55.028999999999996
    - type: ndcg_at_100
      value: 59.244
    - type: ndcg_at_1000
      value: 59.861
    - type: ndcg_at_3
      value: 46.872
    - type: ndcg_at_5
      value: 51.217999999999996
    - type: precision_at_1
      value: 34.936
    - type: precision_at_10
      value: 9.099
    - type: precision_at_100
      value: 1.145
    - type: precision_at_1000
      value: 0.12
    - type: precision_at_3
      value: 21.456
    - type: precision_at_5
      value: 15.411
    - type: recall_at_1
      value: 31.108000000000004
    - type: recall_at_10
      value: 76.53999999999999
    - type: recall_at_100
      value: 94.39
    - type: recall_at_1000
      value: 98.947
    - type: recall_at_3
      value: 55.572
    - type: recall_at_5
      value: 65.525
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
      value: 71.56400000000001
    - type: map_at_10
      value: 85.482
    - type: map_at_100
      value: 86.114
    - type: map_at_1000
      value: 86.13
    - type: map_at_3
      value: 82.607
    - type: map_at_5
      value: 84.405
    - type: mrr_at_1
      value: 82.42
    - type: mrr_at_10
      value: 88.304
    - type: mrr_at_100
      value: 88.399
    - type: mrr_at_1000
      value: 88.399
    - type: mrr_at_3
      value: 87.37
    - type: mrr_at_5
      value: 88.024
    - type: ndcg_at_1
      value: 82.45
    - type: ndcg_at_10
      value: 89.06500000000001
    - type: ndcg_at_100
      value: 90.232
    - type: ndcg_at_1000
      value: 90.305
    - type: ndcg_at_3
      value: 86.375
    - type: ndcg_at_5
      value: 87.85300000000001
    - type: precision_at_1
      value: 82.45
    - type: precision_at_10
      value: 13.486999999999998
    - type: precision_at_100
      value: 1.534
    - type: precision_at_1000
      value: 0.157
    - type: precision_at_3
      value: 37.813
    - type: precision_at_5
      value: 24.773999999999997
    - type: recall_at_1
      value: 71.56400000000001
    - type: recall_at_10
      value: 95.812
    - type: recall_at_100
      value: 99.7
    - type: recall_at_1000
      value: 99.979
    - type: recall_at_3
      value: 87.966
    - type: recall_at_5
      value: 92.268
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
      value: 57.241876648614145
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
      value: 64.66212576446223
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
      value: 5.308
    - type: map_at_10
      value: 13.803
    - type: map_at_100
      value: 16.176
    - type: map_at_1000
      value: 16.561
    - type: map_at_3
      value: 9.761000000000001
    - type: map_at_5
      value: 11.802
    - type: mrr_at_1
      value: 26.200000000000003
    - type: mrr_at_10
      value: 37.621
    - type: mrr_at_100
      value: 38.767
    - type: mrr_at_1000
      value: 38.815
    - type: mrr_at_3
      value: 34.117
    - type: mrr_at_5
      value: 36.107
    - type: ndcg_at_1
      value: 26.200000000000003
    - type: ndcg_at_10
      value: 22.64
    - type: ndcg_at_100
      value: 31.567
    - type: ndcg_at_1000
      value: 37.623
    - type: ndcg_at_3
      value: 21.435000000000002
    - type: ndcg_at_5
      value: 18.87
    - type: precision_at_1
      value: 26.200000000000003
    - type: precision_at_10
      value: 11.74
    - type: precision_at_100
      value: 2.465
    - type: precision_at_1000
      value: 0.391
    - type: precision_at_3
      value: 20.033
    - type: precision_at_5
      value: 16.64
    - type: recall_at_1
      value: 5.308
    - type: recall_at_10
      value: 23.794999999999998
    - type: recall_at_100
      value: 50.015
    - type: recall_at_1000
      value: 79.283
    - type: recall_at_3
      value: 12.178
    - type: recall_at_5
      value: 16.882
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
      value: 84.93231134675553
    - type: cos_sim_spearman
      value: 81.68319292603205
    - type: euclidean_pearson
      value: 81.8396814380367
    - type: euclidean_spearman
      value: 81.24641903349945
    - type: manhattan_pearson
      value: 81.84698799204274
    - type: manhattan_spearman
      value: 81.24269997904105
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
      value: 86.73241671587446
    - type: cos_sim_spearman
      value: 79.05091082971826
    - type: euclidean_pearson
      value: 83.91146869578044
    - type: euclidean_spearman
      value: 79.87978465370936
    - type: manhattan_pearson
      value: 83.90888338917678
    - type: manhattan_spearman
      value: 79.87482848584241
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
      value: 85.14970731146177
    - type: cos_sim_spearman
      value: 86.37363490084627
    - type: euclidean_pearson
      value: 83.02154218530433
    - type: euclidean_spearman
      value: 83.80258761957367
    - type: manhattan_pearson
      value: 83.01664495119347
    - type: manhattan_spearman
      value: 83.77567458007952
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
      value: 83.40474139886784
    - type: cos_sim_spearman
      value: 82.77768789165984
    - type: euclidean_pearson
      value: 80.7065877443695
    - type: euclidean_spearman
      value: 81.375940662505
    - type: manhattan_pearson
      value: 80.6507552270278
    - type: manhattan_spearman
      value: 81.32782179098741
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
      value: 87.08585968722274
    - type: cos_sim_spearman
      value: 88.03110031451399
    - type: euclidean_pearson
      value: 85.74012019602384
    - type: euclidean_spearman
      value: 86.13592849438209
    - type: manhattan_pearson
      value: 85.74404842369206
    - type: manhattan_spearman
      value: 86.14492318960154
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
      value: 84.95069052788875
    - type: cos_sim_spearman
      value: 86.4867991595147
    - type: euclidean_pearson
      value: 84.31013325754635
    - type: euclidean_spearman
      value: 85.01529258006482
    - type: manhattan_pearson
      value: 84.26995570085374
    - type: manhattan_spearman
      value: 84.96982104986162
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
      value: 87.54617647971897
    - type: cos_sim_spearman
      value: 87.49834181751034
    - type: euclidean_pearson
      value: 86.01015322577122
    - type: euclidean_spearman
      value: 84.63362652063199
    - type: manhattan_pearson
      value: 86.13807574475706
    - type: manhattan_spearman
      value: 84.7772370721132
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
      value: 67.20047755786615
    - type: cos_sim_spearman
      value: 67.05324077987636
    - type: euclidean_pearson
      value: 66.91930642976601
    - type: euclidean_spearman
      value: 65.21491856099105
    - type: manhattan_pearson
      value: 66.78756851976624
    - type: manhattan_spearman
      value: 65.12356257740728
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
      value: 86.19852871539686
    - type: cos_sim_spearman
      value: 87.5161895296395
    - type: euclidean_pearson
      value: 84.59848645207485
    - type: euclidean_spearman
      value: 85.26427328757919
    - type: manhattan_pearson
      value: 84.59747366996524
    - type: manhattan_spearman
      value: 85.24045855146915
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
      value: 87.63320317811032
    - type: mrr
      value: 96.26242947321379
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
      value: 60.928000000000004
    - type: map_at_10
      value: 70.112
    - type: map_at_100
      value: 70.59299999999999
    - type: map_at_1000
      value: 70.623
    - type: map_at_3
      value: 66.846
    - type: map_at_5
      value: 68.447
    - type: mrr_at_1
      value: 64.0
    - type: mrr_at_10
      value: 71.212
    - type: mrr_at_100
      value: 71.616
    - type: mrr_at_1000
      value: 71.64500000000001
    - type: mrr_at_3
      value: 68.77799999999999
    - type: mrr_at_5
      value: 70.094
    - type: ndcg_at_1
      value: 64.0
    - type: ndcg_at_10
      value: 74.607
    - type: ndcg_at_100
      value: 76.416
    - type: ndcg_at_1000
      value: 77.102
    - type: ndcg_at_3
      value: 69.126
    - type: ndcg_at_5
      value: 71.41300000000001
    - type: precision_at_1
      value: 64.0
    - type: precision_at_10
      value: 9.933
    - type: precision_at_100
      value: 1.077
    - type: precision_at_1000
      value: 0.11299999999999999
    - type: precision_at_3
      value: 26.556
    - type: precision_at_5
      value: 17.467
    - type: recall_at_1
      value: 60.928000000000004
    - type: recall_at_10
      value: 87.322
    - type: recall_at_100
      value: 94.833
    - type: recall_at_1000
      value: 100.0
    - type: recall_at_3
      value: 72.628
    - type: recall_at_5
      value: 78.428
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
      value: 99.86237623762376
    - type: cos_sim_ap
      value: 96.72586477206649
    - type: cos_sim_f1
      value: 93.01858362631845
    - type: cos_sim_precision
      value: 93.4409687184662
    - type: cos_sim_recall
      value: 92.60000000000001
    - type: dot_accuracy
      value: 99.78019801980199
    - type: dot_ap
      value: 93.72748205246228
    - type: dot_f1
      value: 89.04109589041096
    - type: dot_precision
      value: 87.16475095785441
    - type: dot_recall
      value: 91.0
    - type: euclidean_accuracy
      value: 99.85445544554456
    - type: euclidean_ap
      value: 96.6661459876145
    - type: euclidean_f1
      value: 92.58337481333997
    - type: euclidean_precision
      value: 92.17046580773042
    - type: euclidean_recall
      value: 93.0
    - type: manhattan_accuracy
      value: 99.85445544554456
    - type: manhattan_ap
      value: 96.6883549244056
    - type: manhattan_f1
      value: 92.57598405580468
    - type: manhattan_precision
      value: 92.25422045680239
    - type: manhattan_recall
      value: 92.9
    - type: max_accuracy
      value: 99.86237623762376
    - type: max_ap
      value: 96.72586477206649
    - type: max_f1
      value: 93.01858362631845
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
      value: 66.39930057069995
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
      value: 34.96398659903402
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
      value: 55.946944700355395
    - type: mrr
      value: 56.97151398438164
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
      value: 31.541657650692905
    - type: cos_sim_spearman
      value: 31.605804192286303
    - type: dot_pearson
      value: 28.26905996736398
    - type: dot_spearman
      value: 27.864801765851187
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
      value: 0.22599999999999998
    - type: map_at_10
      value: 1.8870000000000002
    - type: map_at_100
      value: 9.78
    - type: map_at_1000
      value: 22.514
    - type: map_at_3
      value: 0.6669999999999999
    - type: map_at_5
      value: 1.077
    - type: mrr_at_1
      value: 82.0
    - type: mrr_at_10
      value: 89.86699999999999
    - type: mrr_at_100
      value: 89.86699999999999
    - type: mrr_at_1000
      value: 89.86699999999999
    - type: mrr_at_3
      value: 89.667
    - type: mrr_at_5
      value: 89.667
    - type: ndcg_at_1
      value: 79.0
    - type: ndcg_at_10
      value: 74.818
    - type: ndcg_at_100
      value: 53.715999999999994
    - type: ndcg_at_1000
      value: 47.082
    - type: ndcg_at_3
      value: 82.134
    - type: ndcg_at_5
      value: 79.81899999999999
    - type: precision_at_1
      value: 82.0
    - type: precision_at_10
      value: 78.0
    - type: precision_at_100
      value: 54.48
    - type: precision_at_1000
      value: 20.518
    - type: precision_at_3
      value: 87.333
    - type: precision_at_5
      value: 85.2
    - type: recall_at_1
      value: 0.22599999999999998
    - type: recall_at_10
      value: 2.072
    - type: recall_at_100
      value: 13.013
    - type: recall_at_1000
      value: 43.462
    - type: recall_at_3
      value: 0.695
    - type: recall_at_5
      value: 1.139
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
      value: 2.328
    - type: map_at_10
      value: 9.795
    - type: map_at_100
      value: 15.801000000000002
    - type: map_at_1000
      value: 17.23
    - type: map_at_3
      value: 4.734
    - type: map_at_5
      value: 6.644
    - type: mrr_at_1
      value: 30.612000000000002
    - type: mrr_at_10
      value: 46.902
    - type: mrr_at_100
      value: 47.495
    - type: mrr_at_1000
      value: 47.495
    - type: mrr_at_3
      value: 41.156
    - type: mrr_at_5
      value: 44.218
    - type: ndcg_at_1
      value: 28.571
    - type: ndcg_at_10
      value: 24.806
    - type: ndcg_at_100
      value: 36.419000000000004
    - type: ndcg_at_1000
      value: 47.272999999999996
    - type: ndcg_at_3
      value: 25.666
    - type: ndcg_at_5
      value: 25.448999999999998
    - type: precision_at_1
      value: 30.612000000000002
    - type: precision_at_10
      value: 23.061
    - type: precision_at_100
      value: 7.714
    - type: precision_at_1000
      value: 1.484
    - type: precision_at_3
      value: 26.531
    - type: precision_at_5
      value: 26.122
    - type: recall_at_1
      value: 2.328
    - type: recall_at_10
      value: 16.524
    - type: recall_at_100
      value: 47.179
    - type: recall_at_1000
      value: 81.22200000000001
    - type: recall_at_3
      value: 5.745
    - type: recall_at_5
      value: 9.339
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
      value: 70.9142
    - type: ap
      value: 14.335574772555415
    - type: f1
      value: 54.62839595194111
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
      value: 59.94340690435768
    - type: f1
      value: 60.286487936731916
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
      value: 51.26597708987974
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
      value: 87.48882398521786
    - type: cos_sim_ap
      value: 79.04326607602204
    - type: cos_sim_f1
      value: 71.64566826860633
    - type: cos_sim_precision
      value: 70.55512918905092
    - type: cos_sim_recall
      value: 72.77044854881267
    - type: dot_accuracy
      value: 84.19264469213805
    - type: dot_ap
      value: 67.96360043562528
    - type: dot_f1
      value: 64.06418393006827
    - type: dot_precision
      value: 58.64941898706424
    - type: dot_recall
      value: 70.58047493403694
    - type: euclidean_accuracy
      value: 87.45902127913214
    - type: euclidean_ap
      value: 78.9742237648272
    - type: euclidean_f1
      value: 71.5553235908142
    - type: euclidean_precision
      value: 70.77955601445535
    - type: euclidean_recall
      value: 72.34828496042216
    - type: manhattan_accuracy
      value: 87.41729749061214
    - type: manhattan_ap
      value: 78.90073137580596
    - type: manhattan_f1
      value: 71.3942611553533
    - type: manhattan_precision
      value: 68.52705653967483
    - type: manhattan_recall
      value: 74.51187335092348
    - type: max_accuracy
      value: 87.48882398521786
    - type: max_ap
      value: 79.04326607602204
    - type: max_f1
      value: 71.64566826860633
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
      value: 88.68125897465751
    - type: cos_sim_ap
      value: 85.6003454431979
    - type: cos_sim_f1
      value: 77.6957163958641
    - type: cos_sim_precision
      value: 73.0110366307807
    - type: cos_sim_recall
      value: 83.02279026793964
    - type: dot_accuracy
      value: 87.7672992587418
    - type: dot_ap
      value: 82.4971301112899
    - type: dot_f1
      value: 75.90528233151184
    - type: dot_precision
      value: 72.0370626469368
    - type: dot_recall
      value: 80.21250384970742
    - type: euclidean_accuracy
      value: 88.4503434625684
    - type: euclidean_ap
      value: 84.91949884748384
    - type: euclidean_f1
      value: 76.92365018444684
    - type: euclidean_precision
      value: 74.53245721712759
    - type: euclidean_recall
      value: 79.47336002463813
    - type: manhattan_accuracy
      value: 88.47556952691427
    - type: manhattan_ap
      value: 84.8963689101517
    - type: manhattan_f1
      value: 76.85901249256395
    - type: manhattan_precision
      value: 74.31693989071039
    - type: manhattan_recall
      value: 79.58115183246073
    - type: max_accuracy
      value: 88.68125897465751
    - type: max_ap
      value: 85.6003454431979
    - type: max_f1
      value: 77.6957163958641
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

