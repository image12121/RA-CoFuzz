# 公式PAIRアルゴリズム：AutoDL実行設定

## 位置づけ

公式リポジトリ `patrickrchao/JailbreakingLLMs` のcommit
`6379ef705a0fc745530f7d895963510c021b496a`を固定して使用します。

保持するPAIR要素：

- 公式attacker system prompts
- `improvement`と`prompt`を含むJSON攻撃生成
- target responseとJudge scoreを次iterationへ返す反復改善
- 独立した複数conversation streams
- 公式1–10点Judge rubric
- score 10での早期停止
- 会話履歴の切り詰め

接続部分のみ次のように変更します。

- Attacker：DeepSeek API
- PAIR探索Judge：DeepSeek API
- Target：ローカル`Llama-3.2-3B-Instruct`
- 最終評価：既存のStrongJudgeおよびRoBERTa

ローカルTarget生成では、`attention_mask`、`pad_token_id`、`eos_token_id`を
明示してgreedy decodingを行います。これはTransformers 4.46系での未定義padding
警告と、それに伴う生成の不確実性を防ぐためです。

したがって論文上の名称は、`PAIR (official algorithm, budget-matched)`が適切です。

## 質問・target CSV

次のUTF-8 CSVを配置してください。

```text
GPTFuzz-master/datasets/questions/gptfuzzer_q20_seed1234_with_targets.csv
```

列は`index,text,target`、データは既存20問と同一順序の20行です。Smoke testと本実験は
`target`列が存在しない場合、または空欄がある場合に停止します。

## 予算

1問当たり`2 streams × 5 iterations = 最大10 Target queries`です。20問で最大200 queriesです。
公式の早期停止を有効にしているため、実際のquery数は200未満になる場合があります。

## 実行

このディレクトリを次の場所に配置します。

```text
<repository-root>/pair_official_autodl
```

その後、次を実行します。

```bash
cd <repository-root>/pair_official_autodl
chmod +x setup_pair_official.sh smoke_test_pair_official.sh run_pair_official_3runs.sh
bash setup_pair_official.sh
bash smoke_test_pair_official.sh
mkdir -p ../GPTFuzz-master/pair_official_results
bash run_pair_official_3runs.sh 2>&1 | tee ../GPTFuzz-master/pair_official_results/run_all.log
```

`setup_pair_official.sh`は固定コミットだけをHTTP/1.1で最大5回取得します。
GitのTLS通信が失敗した場合は、同じ固定コミットの公式GitHubソースアーカイブへ
自動的に切り替わります。`system_prompts.py`を確認できない限り、セットアップは
成功として扱われません。

## Smoke test

本実験前に1問だけ確認する場合は、次を使います。

```bash
cd <repository-root>/pair_official_autodl
bash smoke_test_pair_official.sh
```

Smoke test完了後、`smoke_summary.json`と末尾ログを確認してから3 runsを開始してください。
