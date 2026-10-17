# Android 控制方案初步调研

日期：2026 年 10 月 16 日  
关联：[手机智能控制器产品提案](手机智能控制器产品提案.md)  
范围：官方接口、项目文档与部分实现检查；候选项目尚未在自有 Android 设备上复测

Android 已有不需要 ESP32 的手机控制方案，也存在使用 ESP32 模拟输入的实现。对当前产品而言，应先比较两条最终可脱离电脑的路线：纯手机 App，以及手机 App 采集画面加外接 HID 输入。电脑加 ADB 则适合作为验证基线。不能因为希望卖硬件，就预先认定 Android 必须经过硬件。

## 已确认的系统能力

Android 的 AccessibilityService 可读取应用暴露的界面节点，调用全局导航动作，并在声明能力和获得用户启用后发送手势。`dispatchGesture` 从 API 24 提供；`takeScreenshot` 从 API 30 提供，需要声明截图能力。因此，Android App 可以在授权范围内取得观察信息并执行部分跨 App 操作，不必天然依赖 ESP32、Root 或电脑。[Android 无障碍服务 API](https://developer.android.com/reference/android/accessibilityservice/AccessibilityService) · [服务启用与配置](https://developer.android.com/guide/topics/ui/accessibility/service)

这条路径取决于目标界面暴露的节点、截图能力和系统限制。官方接口明确包含受保护窗口截图失败的情形；不能把支持无障碍服务等同于可以读取或操作任何界面。

另一种观察通道是 MediaProjection。它可以采集屏幕或单个应用画面，但不提供点击操作。每次开始新的投屏会话需要用户同意；面向 Android 14 及以上时还需符合对应前台服务要求。跨 App Agent 需要检查授权范围是否覆盖整屏。锁屏、用户停止或进程结束等事件会影响采集会话。[Android MediaProjection](https://developer.android.com/media/grow/media-projection)

## 三条候选路线

| 路线 | 怎么看屏幕 | 怎么操作 | 运行时依赖电脑 | 对产品的意义 |
|---|---|---|---|---|
| 纯手机 App | 无障碍节点与截图，必要时另用 MediaProjection | 无障碍节点动作和手势 | 可以不依赖 | 硬件和使用成本最低；需要验证授权体验、后台稳定性与分发渠道 |
| 电脑加 ADB | ADB 截图或相关投屏工具 | ADB 输入或设备端控制服务 | 当前常见框架需要 | 最短实验路径，便于比较模型与任务表现；不直接等于最终消费者体验 |
| 手机 App 加 ESP32 | App 使用系统允许的截图或屏幕采集接口 | App 下发命令，ESP32 模拟 USB 或蓝牙键鼠 | 目标是不依赖 | 有望减少对无障碍手势注入的依赖；仍需系统允许采集画面，硬件不能替代所有权限 |

ADB 需要用户开启调试并授权电脑，不能将开发调试通道当成消费者默认拥有的权限。[Android ADB 文档](https://developer.android.com/tools/adb)

第三条是待实现、待实测的设计候选。可先采用手机 App 经 BLE 给 S3 发少量命令、S3 经 USB HID 操作手机的分工；画面直接由 App 上传模型，不经过板子。若改用 BLE HID，又需要验证同一手机与板子的 App 通信、系统键鼠连接如何共存。

## 找到的相关项目

| 项目 | 已有路线 | 适合参考什么 | 尚不能据此认定什么 |
|---|---|---|---|
| [Juyu 手机 Agent](https://github.com/goehou/Juyu-phone-agent) | Android App、无障碍截图与手势、用户自行配置模型 API；项目要求 Android 11 及以上 | 最接近“装 App 后在手机上运行”的无硬件对照组；配置与源码包含截图和手势能力 | 尚未在我们的设备上验证，也不能把项目支持某模型名等同于该模型支持视觉或稳定完成任务 |
| [AutoJs6](https://github.com/SuperMonster003/AutoJs6) | 手机上的 JavaScript 自动化工具，包含无障碍控件操作、截图与脚本运行 | 验证手机独立执行、控件操作和截图能力 | 它是自动化工具基础，不等于已经完成的通用云端 Agent 产品 |
| [Open AutoGLM](https://github.com/zai-org/Open-AutoGLM) | 电脑 Python Agent，Android 使用 ADB；已有 HarmonyOS HDC 后端 | 快速建立无硬件实验基线，以及参考任务、截图和动作闭环 | 默认模型与动作格式有其适配要求，不能仅修改地址就承诺 DeepSeek 可直接替换 |
| [Mobilerun 原 DroidRun](https://github.com/droidrun/mobilerun) | 电脑端 Agent 加 Android Portal App，结合无障碍树、截图与动作接口 | 对比纯截图与语义节点两种观察方式 | Portal 是设备端组件，当前本地框架的 Agent 仍跑在电脑；不能称为完整手机独立产品 |
| [ESP32 Android Agent](https://github.com/shenlan-ai/esp32-android-agent) | 电脑取得 ADB 截图，使用 ESP32 BLE HID、S3 USB HID 或 ADB 输入 | 与当前 iPhone 项目思路非常接近，并提供无板子对照模式 | 作者记录的实机证据主要是经典 ESP32 加 Redmi Note14 的 Android 15；S3 手机端验收仍待完成，不代表我们的 S3 已兼容 |

Juyu 的权限配置声明了 `canPerformGestures`、`canRetrieveWindowContent` 和 `canTakeScreenshot`，其路线并非只停留在概念描述。[权限配置源码](https://github.com/goehou/Juyu-phone-agent/blob/main/app/src/main/res/xml/accessibility_service_config.xml)

ESP32 Android Agent 仍然靠 ADB 看屏幕，尚未提供配套 Android App，因此它可以帮助验证外接输入，却不能直接满足“扫码安装 App 后无需电脑”的目标。项目同时将硬件实测范围与编译、模拟测试区分开来。[项目兼容性说明](https://github.com/shenlan-ai/esp32-android-agent/blob/main/docs/COMPATIBILITY.md)

本次记录的版本基线：Open AutoGLM 为 `86f5538`，Mobilerun 为 `3018561`，AutoJs6 为 `ed3eb10`，Juyu 为 `7810db3`，ESP32 Android Agent 为 `529c05b`。源码许可依次显示为 Apache 2.0、MIT、MPL 2.0、Apache 2.0 和 MIT。采用任何项目之前仍需核对实际纳入的代码、依赖与对应分发要求。

## 商店分发与系统能力需要分别判断

Google Play 当前政策禁止利用 Accessibility API 让 App 自主发起、规划并执行动作或决策；明确区分了确定性、人工预定义的规则自动化。经验证、以帮助残障用户为核心目的的辅助工具有特定例外，面向大众的通用助手不能仅靠声明该标志就适用例外。[Google Play 无障碍 API 政策](https://support.google.com/googleplay/android-developer/answer/10964491?hl=en)

因此，“手机上能运行无障碍 Agent”和“能按这个产品形态在 Google Play 分发”必须分别验证。这是 Google Play 的渠道规则，不能扩大成所有 Android 系统都禁止该技术，也不能据此认定国内其他渠道一定允许。

若外接硬件只承担输入、App 改用 MediaProjection 看屏幕，可以研究减少对 Accessibility API 的依赖；这只是架构方向，不能推出已经符合全部商店规则或能够免授权使用。

## 鸿蒙需要独立验证

Open AutoGLM 已提供 `phone_agent/hdc` 后端，源码通过 HDC 调用截图工具及 `uitest uiInput` 完成操作，证明存在开发调试路线的实现。[HDC 输入实现](https://github.com/zai-org/Open-AutoGLM/blob/main/phone_agent/hdc/device.py) · [HDC 截图实现](https://github.com/zai-org/Open-AutoGLM/blob/main/phone_agent/hdc/screenshot.py)

这不证明普通消费者安装一个鸿蒙 App 后就拥有同样权限。后续必须确认具体设备与系统代际，分别核对应用安装格式、屏幕采集、跨应用操作、外接 HID 识别和后台运行。Android APK 的能力不能直接外推到鸿蒙原生应用。

## 对 USB-C 硬件形态的判断

ESP32-S3 的原生 USB OTG 为 Full Speed，标称总线速率 12 Mbps，可实现 HID 等 USB 设备。这足以说明它可承担键鼠输入，但不能把它当作手机完整 USB-C 功能的透明中继。[乐鑫 USB OTG 文档](https://docs.espressif.com/projects/esp-iot-solution/en/release-v2.0/usb/usb_overview/usb_otg.html)

带充电、数据和视频输出的 USB-C 扩展坞通常涉及独立的 Hub、PD 供电协商和信号切换路径。TI 的扩展坞参考方案就将这些电路分别列出。[TI USB-C Mini Dock 参考方案](https://www.ti.com/tool/USB-CTM-MINIDK-EVM)

据此，建议把未来硬件理解为“扩展坞中集成控制模块”：S3 是输入设备的一部分，原接口其他能力由相应扩展电路承载。第一阶段先测试现成带充电功能的 Hub 加 S3 是否能够同时工作，再决定是否设计一体化电路板。具体手机能否边充电边使用控制模块仍需实测。

外接输入和屏幕获取始终是两个问题。例如 scrcpy 的 OTG 模式不需要 ADB 就能模拟键鼠，但该模式不提供视频或音频。这不能直接作为无授权看屏幕的方案。[scrcpy OTG 说明](https://github.com/Genymobile/scrcpy/blob/master/doc/otg.md)

## 下一轮实验建议

先完成 iPhone 16 Pro Max 的 ESP32 控制验证，再在已有 Android 设备上做同任务比较。以下是建议的验收顺序，尚未代表产品路线已经选定。

| 实验 | 最小输入 | 观察结果 | 影响的决定 |
|---|---|---|---|
| Android 无硬件基线 | 一台 Android 手机、ADB 授权、电脑 | 截图、点击、滑动、文本输入是否通畅；模型是否完成短任务 | 分离模型问题与硬件问题 |
| 手机独立软件路线 | Android App、用户启用必要服务、模型 API | 离开电脑后能否完成任务；后台、横竖屏、重启后是否可恢复 | 是否需要外接硬件 |
| 手机独立硬件路线 | 屏幕采集 App、S3、USB 数据连接、命令通道 | 采集与 HID 能否同时稳定工作，是否明显减少配置或提高成功率 | 硬件在 Android 上的必要价值 |
| 同时充电 | 已通过输入测试的手机与 S3、合适的现成 Hub 和充电器 | 供电、充电与输入是否同时维持，插拔后是否能恢复 | 是否进入 USB-C 扩展硬件设计 |
| 新用户首次使用 | 未参与开发的测试者、完整安装和引导流程 | 完成首次任务所需步骤、时间、求助次数 | 是否接近消费者交付 |

统一记录任务结果、总耗时、操作偏差、首次设置成本、断线恢复、重启恢复和需要人工接管的次数。一个成功演示不等于稳定性结论。

当前倾向是：保留 App 加硬件的产品愿景，对 Android 先建立无硬件对照组；如果硬件能在目标设备上显著降低使用摩擦或提升可靠性，再将它作为 Android 标配。iOS 与 Android 可以共享任务体验和云端服务，而不强求相同的观察与输入实现。
