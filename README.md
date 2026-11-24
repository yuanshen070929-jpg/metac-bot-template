# YuanShenbot · Metaculus FutureEval 预测机器人

这是参加 [Metaculus FutureEval](https://www.metaculus.com/futureeval/participate/) 机器人锦标赛用的预测 bot，在 Metaculus 官方模板 [metac-bot-template](https://github.com/Metaculus/metac-bot-template) 的基础上改成了"多家模型集成"的版本。

官方模板的文件（`main.py`、`main_with_no_framework.py` 等）原样保留，方便对照。新写的代码在 `ensemble_bot/` 和 `run_bot.py`，GitHub Actions 已经改成运行新 bot。官方模板原来的英文说明见 [上游 README](https://github.com/Metaculus/metac-bot-template#readme)。

---

## 它怎么预测一道题

```
题目 ──► ① 研究（几路来源并行）──► ② 三家模型各自独立预测 ──► ③ 聚合 ──► ④ 发布预测 + 私密评论
```

1. **研究**：下面几路同时跑，任何一路失败都不影响其他：
   - **AskNews 新闻检索**：比赛参与者可以免费用，有 key 才会开启；
   - **联网搜索模型**：默认 `gpt-6.1-terra:online`（OpenAI 原生联网搜索）和 Perplexity `sonar-pro`；
   - **直接读判定数据源**：很多题会写明"以某某网页的数据为准"。bot 会把这些网页抓下来，让一个便宜的小模型只摘出跟判定有关的数字和日期；
   - **时间信息由代码计算**：今天的日期、截止时间、结题日期、还剩多少天，不让模型自己猜。
2. **预测**：同一份研究交给三个不同厂商的前沿模型，各自独立推理。默认是 OpenAI `gpt-6.2-sol`、Anthropic `claude-opus-5.5`、Google `gemini-4-argon`。Gemini 调用失败时会自动换成 `gemini-3.8-flash` 补位。只要有一个模型成功，这道题就照样发布。
3. **聚合**：沿用 forecasting-tools 自带的方法，二元题取中位数，多选题取平均，数值题对累积分布逐点取中位数。
4. **评论**：每道题会自动发一条私密评论，里面有研究内容和每个模型完整的推理过程，并标明是哪个模型写的。拿奖要求评论如实反映 bot 的推理，这一步是自动完成的。

### 和官方模板相比改了什么

| | 官方模板 | 这个 bot |
|---|---|---|
| 预测模型 | 同一个模型重复 5 次 | 三家厂商的模型各 1 次。不同厂商犯的错相关性更低，取中位数更稳 |
| 研究 | 一个来源 | 新闻 + 联网搜索 + 直接读判定数据源，并行执行 |
| 时间信息 | 只告诉模型"今天几号" | 代码算好开题、截止、结题时间和剩余天数 |
| 提示词 | 现状、正反情景 | 判定机制 → 现状 → 基准率（附时间缩放公式）→ 证据 → 情景 → 校准检查 |
| 成本控制 | 无 | 每次运行最多处理 N 题；先做快截止的题；马上截止的题直接跳过 |
| 测试 | 无 | 30 个离线测试，用假模型跑完整流程，不联网也不花钱 |

最终答案的格式和解析逻辑沿用官方模板，Metaculus 每个赛季都会维护这部分。

---

## 第一次部署

1. **Fork 仓库**：已经完成，就是这个仓库。仓库要保持公开：公开仓库的 GitHub Actions 免费且不限时长，私有仓库每月只有 2000 分钟，bot 每 20 分钟跑一次会超额。
2. **添加密钥**：打开 Settings → Secrets and variables → Actions → **New repository secret**，逐个添加：
   - `METACULUS_TOKEN`：**bot 账号**的 token，不是你个人账号的。在 Metaculus 设置 → My Forecasting Bots 里找；
   - `OPENROUTER_API_KEY`：在 openrouter.ai → Keys 新建，建议给这个 key 设一个额度上限；
   - （可选）AskNews 的密钥：在 my.asknews.app 点 Create key。如果给的是一个 API key，就存成 `ASKNEWS_API_KEY`；如果给的是 Client ID 和 Secret，就分别存成 `ASKNEWS_CLIENT_ID` 和 `ASKNEWS_SECRET`。加了就会自动启用。
3. **启用 Actions**：打开仓库的 Actions 页面，点 "I understand my workflows, go ahead and enable them"。Fork 出来的仓库默认不运行定时任务，这一步必须手动点。
4. **跑一次测试**：Actions → **Test Bot** → Run workflow。它会先跑离线测试，再在 [bot-testing-area](https://www.metaculus.com/tournament/bot-testing-area/) 的练习题上真实预测并发帖（练习题不计分）。查看结果的方法：用你本人的账号登录 Metaculus → Settings → My Forecasting Bots → Switch to bot account，然后打开日志里列出的题目链接。
5. **正式运行**："Forecast on new AI tournament questions" 每 20 分钟自动跑一次，已经预测过的题会跳过，没有新题的时候不花钱。

## 可选设置

在 Settings → Secrets and variables → Actions → **Variables** 里添加仓库变量，不用改代码。都不设就用默认值。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RUN_MODE` | `tournament` | `tournament` = 赛季主赛 + MiniBench；`main` = 只跑主赛；`minibench` = 只跑 MiniBench |
| `BOT_PROFILE` | `standard` | `standard`：gpt-6.2-sol + claude-opus-5.5 + gemini-4-argon；`budget`：gpt-6.1-terra + claude-sonnet-5.5 + gemini-3.8-flash |
| `FORECASTER_MODELS` | （按 profile） | 自定义参与集成的模型，逗号分隔的 OpenRouter 模型名；`a\|b` 表示 a 失败时用 b |
| `RESEARCH_MODELS` | （按 profile） | 联网搜索模型，逗号分隔；填 `none` 表示关闭 |
| `SAMPLES_PER_MODEL` | `1` | 每个模型对每道题预测几次 |
| `REASONING_EFFORT` | standard 为 `high`，budget 为 `medium` | 模型的推理强度：`low` / `medium` / `high` / `none` |
| `MAX_QUESTIONS_PER_RUN` | `15` | 每次运行最多处理几道题，剩下的留给下一次 |
| `USE_ASKNEWS` | 有 key 就开 | 强制开（`true`）或关（`false`）AskNews |
| `TOURNAMENT_IDS` | （库里的当前赛季） | 手动指定主赛和 MiniBench 的 ID 或 slug，逗号分隔；换赛季时有用 |

## 成本（粗略估算）

- `standard`：每题约 **$0.4–0.6**；`budget`：每题约 **$0.15–0.25**。这是按当前 OpenRouter 价格粗算的，实际花费以 OpenRouter 后台为准。
- MiniBench 每两周一轮，每轮约 60 题；赛季主赛每季 300–400 题。
- 拿到 Metaculus 的免费额度之前，建议设 `RUN_MODE=minibench`，必要时再加 `BOT_PROFILE=budget`。
- 一定要在 OpenRouter 上给 key 设额度上限。

## 比赛规则提醒

- 每道题都要有评论，并如实反映推理过程。bot 会自动发。
- 参赛表要填，每个赛季结束后的 bot 问卷也必须填，否则拿不到奖。
- bot 必须全自动运行。按官方以往的规则，不能先看 bot 在正在进行的比赛题上怎么预测，再据此改 bot 去"修"这些题，这算 human in the loop。要改进 bot，请用已经结题的题或练习题。
- 以 3 人以上营利公司的名义做的 bot 不能拿奖（全部开源的除外）。以个人名义参赛没问题。

## 换赛季要做的事（每 4 个月一次）

1. 更新 forecasting-tools，拿到新赛季的比赛 ID：`uv lock --upgrade-package forecasting-tools`，然后提交 `uv.lock`。也可以不更新，直接设 `TOURNAMENT_IDS` 变量。
2. 新模型发布很快。先去 OpenRouter 的模型列表看看三家有没有更新的模型，再改 `FORECASTER_MODELS`。
3. 想同步官方模板的更新时，注意我们改过 workflows、README、`.env.template`、`pyproject.toml` 和 `uv.lock`，GitHub 上的 "Sync fork" 可能会提示冲突，在本地合并更稳妥。

## 可靠性

- GitHub 的定时任务高峰期会延迟，偶尔还会漏跑。以往赛季的主赛题目每 2 小时放出一道，每道只开放 1.5 小时，漏跑就会错过题目。前几天请多看看 Actions 的运行记录。如果漏得多，可以在 cron-job.org 上每 15 分钟调用一次 GitHub API，手动触发这个 workflow（需要一个只授权本仓库 Actions 读写权限的 fine-grained token）。
- 公开仓库如果 60 天没有任何提交，GitHub 会自动停用定时任务。赛季中记得偶尔提交一下，或者到 Actions 页面重新启用。
- 某次运行里有题目失败时，这次运行会标红，GitHub 会发邮件提醒。失败的题下一次运行会重试，但每次重试都要花钱，所以看到连续标红请及时检查。

## 本地运行和测试

```bash
uv sync
uv run pytest -q                                          # 离线测试，不联网不花钱
cp .env.template .env                                     # 填入 METACULUS_TOKEN 和 OPENROUTER_API_KEY
uv run python run_bot.py --mode test_questions --dry-run  # 在练习题上预测，但不发布
uv run python run_bot.py --mode minibench                 # 正式预测 MiniBench
```

## 代码结构

| 文件 | 作用 |
|---|---|
| `run_bot.py` | 入口：拉取题目、筛选、预测、汇总 |
| `ensemble_bot/bot.py` | 集成 bot，继承官方 `FallTemplateBot2026` |
| `ensemble_bot/research.py` | 并行研究：AskNews、联网搜索、判定数据源 |
| `ensemble_bot/sources.py` | 从题目里提取网址、抓取网页、HTML 转文本 |
| `ensemble_bot/prompts.py` | 所有提示词 |
| `ensemble_bot/config.py` | 从环境变量读取设置；`standard` / `budget` 两套模型配置 |
| `ensemble_bot/dates.py`、`selection.py` | 时间信息计算；每次运行挑选哪些题 |
| `tests/` | 离线测试 |
| `main.py` | 官方模板原版（未改动，作参照） |

## 有用的链接

- 参赛页面：https://www.metaculus.com/futureeval/participate/
- 资源页（规则、开源 bot 列表）：https://www.metaculus.com/notebooks/38928/futureeval-resources-page/
- Metaculus Discord 的 build-a-forecasting-bot 频道：https://discord.gg/metaculus
