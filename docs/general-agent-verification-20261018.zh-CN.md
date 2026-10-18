# 通用设备控制验证记录：动作校验、三次重启与工具栏探索

日期：2026-10-18  
范围：iPad（投屏报告型号 `iPad16,8`，竖屏 752×1080）上的通用闭环验证  
原则：**不为 Freeform 或“画笑脸”写固定坐标、按钮顺序或任务专用分支。** 本文只记录通用机制、真机结果和模型能力边界。

## 1. 结论摘要

| 项目 | 结果 |
|---|---|
| 设备基准（baseline） | **通过**，跨度 w=1193、h=1721，4 个留出点最大轴向误差 0.51% |
| 正向对照“回到主屏幕” | **通过**，动作校验 `validated=true`，完成校验 `verified=true` |
| “打开白板画一个笑脸” | **失败**，22 步、3 轮尝试后如实 `fail`；校验和重启机制均生效，未出现假完成 |
| 左右坐标是否反了 | **没有反**；baseline 的 x 轴留出点 0.25→0.25197、0.75→0.75441，方向与目标一致 |
| 批量执行 / 无校验 done | **已修复**，单轮只接受一个 DO，done 必须有执行后截图上的独立判定 |
| 主要剩余瓶颈 | `deepseek-flash` 对 iPad 工具栏图标的语义识别：把 Share/Insert 认成画笔，而不是坐标或宽度问题 |

## 2. 环境与版本

- 服务端入口：`bash scripts/run-device-baseline.sh`；数据目录 `.local/device-baseline/`。
- Web/API：`http://127.0.0.1:8890`；设备 TCP：`7300`；AirPlay 接收名：`iPhone Agent`。
- 屏幕帧：752×1080，宽高比 0.6963；UxPlay 1.73.7 + 项目内 GStreamer。
- 当前源码提交：`a2d9547`（另有 `c9c150e`、`7490011`）。
- 测试：`84 passed, 1 skipped`，ruff 通过。
- 投屏握手只提供设备型号和来源摘要；系统版本没有由接口单独暴露，本文不猜测。

## 3. 已落地的通用机制

| 机制 | 位置 | 行为 |
|---|---|---|
| 单动作回合 | `server/hidagent/agent.py` `parse_actions` | 一个 DO 行；多 DO 直接拒绝并让 Actor 重答 |
| 点击前 grounding | `server/hidagent/grounding.py` | 在当前截图里定位目标；顶部工具栏 tap 使用整条工具栏带 |
| 步骤校验 | `Agent._validate_action` | 独立 step verifier 对照 Actor 声明的意图和当前截图；不匹配就给出实际看到的内容 |
| 三次失败重启 | `Agent._step_check_failed`、`Agent._restart` | 同一轮累计 3 次失败（含 grounding 失败）→ Home 键回主屏幕，重新规划；最多 3 轮 |
| 失败摘要 | `Agent._failure_summary` | 重启后的 Planner 收到上一轮“动作 → 实际结果”，避免无依据地重复 |
| 工具栏邻位探索 | `Agent._next_actions` 的 `explore` 标记 | 换到不同工具栏 x 时跳过语义 grounding，直接点该邻位并观察结果 |
| 完成校验 | `Agent._verify` | done 必须由 verifier 用执行后的新截图确认；未确认则记 hiccup 并重规划 |

## 4. 验证流程（可回放）

1. 启动服务端：

   ```bash
   cd /Users/Luaphes/Documents/ChatGPT/esp控制手机
   bash scripts/run-device-baseline.sh
   ```

2. 连接 ESP32 并选择屏幕镜像，确认：

   ```bash
   curl -s http://127.0.0.1:8890/api/status
   ```

   期望 `device=true, screen=true`。

3. 建立 baseline。连接/投屏稳定后会自动测量；失败时手动重试：

   ```bash
   curl -s -X POST http://127.0.0.1:8890/api/calibrate/start
   ```

   期望 `baseline.phase=ready, calibrated=true`，且 `checks` 中每个 `error` 小于 0.018。

4. 执行任务：

   ```bash
   curl -s -X POST -H 'Content-Type: application/json' \
     -d '{"text":"打开白板画一个笑脸"}' \
     http://127.0.0.1:8890/api/agent
   ```

5. 观察；需要人工停止时：

   ```bash
   curl -s -X POST http://127.0.0.1:8890/api/agent/stop
   ```

6. 回放轨迹：

   - 目录：`.local/device-baseline/traces/<YYYYMMDD-HHMMSS>/`
   - `steps.jsonl` 是结构化事件；偶数编号 JPEG 是 Actor/verifier 看到的整屏，奇数编号 1000×1000 JPEG 是 grounder 的裁剪图。
   - 关键字段：`validated`、`validation`、`validate_s`、`explore`、`verified`、`verification`、`restart`、`fail`。

## 5. 本轮结果

### 5.1 Baseline

2026-10-18 08:59 手动重试通过：

| 目标 | 实测 | 轴向误差 |
|---|---|---|
| (0.50, 0.50) | (0.50297, 0.50150) | 0.30% |
| (0.75, 0.80) | (0.75441, 0.80193) | 0.44% |
| (0.82, 0.99) | (0.82505, 0.99088) | 0.51% |
| (0.25, 0.20) | (0.25197, 0.19724) | 0.28% |

跨度 w=1193、h=1721。08:57 的自动尝试曾在 `measure 5/5` 只拿到 2 个观测而失败，手动重试通过；说明测量阶段仍受画面/指针对比度影响，需要重试策略。

### 5.2 正向对照：“回到主屏幕”通过

轨迹：[../.local/device-baseline/traces/20261018-090525/steps.jsonl](../.local/device-baseline/traces/20261018-090525/steps.jsonl)

- `swipe`：`validated=true`，校验器确认已回到主屏幕；
- `done`：`verified=true`，完成校验理由为“屏幕显示 iPad 主屏幕、应用图标、小组件和 Dock”。

这条证明：单动作、步骤校验、完成校验在普通任务上不会误杀。

### 5.3 “打开白板画一个笑脸”失败

轨迹：[../.local/device-baseline/traces/20261018-085925/steps.jsonl](../.local/device-baseline/traces/20261018-085925/steps.jsonl)

- 22 步、3 轮尝试后 `fail`，最后原因是 `three step checks failed in attempt 3; no restart attempts left`；
- 没有出现假 `done`；
- 校验器实际抓到：
  - tap `823,46` → 打开 **分享弹窗**，不是画笔；
  - tap `620,46` → 打开 **附件/Insert 菜单**（Browse Content Hub、Choose Photo…），不是画笔；
  - 一次 `explore=true` 的 tap 确实直接点了 `620`（没有被 grounding 拉回），但 620 本身就是错误图标；
- 重启后 Planner 收到失败摘要，但仍重复“点工具栏画笔图标”；随后随机点击使应用状态漂移到 App Store、照片、搜索等页面，第三轮在错误页面上耗尽尝试。

### 5.4 关键历史轨迹

| 轨迹 | 结论 |
|---|---|
| `20261018-003024` | 空白画布上的 tap 被 grounder 判为 NOT_FOUND，0 动作失败；随后已允许空白区域 |
| `20261018-010343` | 加入单动作/步骤校验后，校验/重启生效，但 grounding 异常被普通错误计数提前终止 |
| `20261018-011734` | 邻位提示后 Actor 提出 `x=540`，但 grounding 又拉回 `x=620`；随后用 `explore` 直接点击修复 |
| `20261018-085925` | 最新验证：单动作、grounding、步骤校验、三次重启、失败摘要、explore 全部执行；任务仍因模型图标识别失败 |

## 6. 发现

1. **不是宽度或横竖屏。** baseline 的 x/y 留出点与目标同向且误差小于 0.51%；同一套坐标能把点击准确送到 `823`、`620` 等位置，失败是“点到的是哪个图标”，不是“坐标偏了多少”。
2. **左右没有反。** 0.25 实测 0.25197，0.75 实测 0.75441；若镜像会互换。模型持续选择右侧/中间的 Share、Insert，是语义误判。
3. **工具栏语义识别是当前主瓶颈。** 整条工具栏放大后，grounder 仍返回 `x=620`（Insert）；Actor 也反复认为 `x=620` 是画笔。真正的绘图入口在左侧（标题后的第一个图标），但模型没有建立这个映射。
4. **通用安全机制有效。** 多 DO 被拒绝；步骤不匹配会被独立校验器抓住；三次失败会回主屏幕重开；done 不会被空白画布骗过。正向对照通过。
5. **邻位探索需要由 harness 保证，而不是靠 Actor 自觉。** 最新 `explore` 只能保证“Actor 换 x 时不再被 grounding 拉回”；如果 Actor 继续提同一个 `x=620`，邻位搜索不会发生。下一步应做 harness 侧的候选序列，而不是继续加提示词。
6. **重启会丢失应用上下文。** 回主屏幕后，Planner 需要重新打开应用；当前模型常把它做成“随便点一个图标/打开 App Store”，导致状态漂移。恢复策略需要固定“先回到可复现的任务入口”这一通用步骤。
7. **baseline 测量存在偶发低观测。** 白色画布、浅色背景上更容易只拿到 2 个观测；手动重试可过，但应自动换点重试，而不是让用户点第二次。

## 7. 建议

按通用性排序：

1. **换更强的视觉模型（最直接）。** 当前 Actor 和 Verifier 都是 `deepseek-flash`；它读文字/菜单可靠，但区分工具栏图标不可靠。换模型不需要任何 App 专用代码，是当前最可能真正解决“画笔在左、模型点右”的办法。
2. **Harness 侧系统化工具栏扫描。** 不依赖 Actor 选 x：某次工具栏 tap 校验失败后，按固定偏移依次尝试左/右相邻图标（例如 ±8% 屏宽），每次都用 step verifier 判断打开了什么；探索 tap 跳过语义 grounding（`explore` 已有）。扫描游标跨重启保留，避免重复。
3. **重启时保留“可复现入口”。** 回主屏幕后先通过通用搜索/最近使用重新打开目标应用，再执行新计划；失败摘要继续传给 Planner，避免它把“回主屏”理解成“随便点”。
4. **baseline 测量自动换点。** 5 个测量点里可观测少于 3 个时，自动在同高度带换备选点重试；不要直接失败等用户手动点。
5. **验收指标。** 记录成功/失败、误报完成、人工接管、重启次数；任何一次“空白画布 done”都判该轮不通过。当前没有误报完成，但样本仍少。

## 8. 当前状态

- 服务端运行中，`device=true, screen=true, calibrated=true, baseline.phase=ready`；
- Agent 空闲；最后一次正向对照已回到主屏幕；
- 工作树干净，最新提交 `a2d9547`；
- `.local/device-baseline/traces/` 为本地证据，不随 Git 提交；本文可作为验证过程与回放入口。

## 9. 证据索引

- 通用控制标准：[general-control-standard.zh-CN.md](general-control-standard.zh-CN.md)
- 设备基准说明：[device-baseline.zh-CN.md](device-baseline.zh-CN.md)
- 正向对照轨迹：[20261018-090525/steps.jsonl](../.local/device-baseline/traces/20261018-090525/steps.jsonl)
- 最新画笑脸轨迹：[20261018-085925/steps.jsonl](../.local/device-baseline/traces/20261018-085925/steps.jsonl)
- 早期空白画布失败：[20261018-003024/steps.jsonl](../.local/device-baseline/traces/20261018-003024/steps.jsonl)
- Grounding 拉回问题：[20261018-011734/steps.jsonl](../.local/device-baseline/traces/20261018-011734/steps.jsonl)
- 服务端运行日志：`../.local/device-baseline-server.log`
