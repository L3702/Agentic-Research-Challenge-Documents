# E1 词典与停用词来源

## 词典

| 名称 | 来源 | 许可 | 本地缓存 |
|------|------|------|----------|
| NTUSD 简体中文情感词典 | `ntunlplab/NTUSD`（MIT）；简体文件来自 `ppzhenghua/SentimentAnalysisDictionary` 公开汇总仓库 | 原始仓库 MIT；简体汇总仓库未标注许可，仅用于学术研究 | `_temp/e1_dictionaries/ntusd_simplified_positive.txt`、`ntusd_simplified_negative.txt` |
| 大连理工大学情感词汇本体 | `yizhanmiao/DLUT-Emotionontology` | 学术研究许可，商业用途需联系 `irlab@dlut.edu.cn` | `_temp/e1_dictionaries/dlut_emotion.csv` |
| 自制 MPR 宏观金融领域词典 v1 | 本实验自建 | 无外部许可限制 | `experiments/E1/dictionaries/mpr_domain_dictionary.csv` |

## 停用词

| 名称 | 来源 | 许可 | 本地缓存 |
|------|------|------|----------|
| stopwords-zh | `stopwords-iso/stopwords-zh` | MIT | `_temp/e1_dictionaries/stopwords_zh.txt` |

## 版本固定

- 抓取日期：2026-11-18
- jieba：0.42.1
- NTUSD 原始仓库 commit：`main` 分支抓取
- DLUT 仓库 commit：`master` 分支抓取
- 自制词典版本：`self_made_mpr_domain_v1`

## 说明

- NTUSD 原始仓库为繁体中文 Big5 编码；本实验使用公开汇总仓库提供的简体中文 UTF-16 版本，并去除了正负词表中的重叠词。
- DLUT 按词语聚合，若同一词语同时具有正向和负向极性，则作为歧义词剔除；中性词不参与评分。
- 自制词典只收录立场明确、语境依赖较低的宏观金融词，允许后续版本修订。
