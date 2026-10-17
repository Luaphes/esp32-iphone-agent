# iOS 控制验证记录

日期：2026 年 10 月 17 日

本轮已跑通 `legacy` 电脑常驻架构下的 iPhone 16 Pro Max 手动控制和 DeepSeek 短任务。源码基线为 `fc8458c`，本地分支为 `luaphes/ios-validation`。这是一轮真实设备闭环验证，尚不代表复杂任务或多机型的稳定性。

## 连接方式

刷写时，ESP32-S3 通过 USB 连接 Mac。运行时，ESP32-S3 改接 iPhone 的 USB-C 口，作为 USB 键盘和鼠标；电脑通过局域网 Wi-Fi 向板子发送命令，手机通过 AirPlay 把屏幕投给电脑。当前固件没有使用蓝牙。

手机、电脑和板子需要处于可互通的局域网；ESP32 使用 2.4 GHz Wi-Fi，手机和电脑可以使用同一路由器下可互通的其他频段。当前电脑局域网地址为 `192.168.31.214`，控制器连接端口为 `7300`。

## 已识别硬件

| 项目 | 实际读取结果 |
|---|---|
| 芯片 | ESP32-S3 QFN56，revision v0.2 |
| Flash | 4 MB，Quad，3.3 V |
| 内置 PSRAM | 2 MB |
| 刷写接口 | 原生 USB Serial JTAG |
| 初始串口 | `/dev/cu.usbmodem14301` |
| 本次固件 | Wi-Fi TCP 命令通道，USB HID 键鼠，USB CDC 调试日志 |

这些信息识别的是芯片和存储规格，不足以判定开发板的商品型号。本次将构建参数从上游默认 8 MB 改为 4 MB，采用默认 4 MB 分区；不应对这块板子直接使用新版 main 的默认 8 MB Flash 和 OPI PSRAM 配置。

刷写前的完整 4 MB Flash 已保存在 `.local/backups/esp32s3-20261017-original.bin`，该目录不提交到 Git。备份 SHA256：`0dc8e79f4bbd9837d8f17e9e6050eed696530921d1fd215ab24a59458215decc`。

## 电脑端环境

- Arduino CLI 1.5.1，ESP32 Arduino Core 3.3.12；核心与编译缓存位于项目 `.tools/`。
- Python 3.14.6，项目虚拟环境为 `server/.venv/`，已补齐本机 SOCKS 代理所需的 `socksio`。
- UxPlay 1.73.7，源码提交 `df67c21`；本机构建位于 `.tools/uxplay/`。为定位握手问题，日志输出增加了即时刷新。
- GStreamer 1.26.11 官方 macOS Universal 运行库与 SDK，下载文件通过官方 SHA256 校验，解包至项目 `.tools/GStreamer.framework/`。
- UxPlay 使用项目内选定的视频处理插件及软件解码；未安装完整 Xcode。
- libplist 2.8.0、pkgconf 3.0.8 已安装；CMake 复用现有 PlatformIO 所附的 3.30.2。

本机启动入口：

```bash
scripts/run-local.sh
```

网页控制台为 [http://127.0.0.1:8890](http://127.0.0.1:8890)，仅监听本机。ESP32 的 TCP 服务监听局域网；AirPlay 接收名称为 `iPhone Agent`。

固件编译入口：

```bash
hardware/flash.sh --debug --build-only
```

`hardware/flash.sh` 支持指定串口，且不会在存在多块设备时随意挑选第一块；其他存储规格可以通过 `ESP32_FQBN` 覆盖。再次刷写前须把目标板接回电脑，并确认对应串口。

## 本轮验证状态

| 检查项 | 状态与证据 |
|---|---|
| 芯片和 Flash 识别 | 已完成，esptool 读取 |
| 原固件备份 | 已完成，4 MB 文件与 SHA256 已记录 |
| 4 MB 固件编译 | 已通过；程序 964009 字节，静态全局内存 78688 字节 |
| 固件刷写 | 已通过，写入后哈希校验成功 |
| ESP32 连接电脑 | 已通过，TCP 接收端收到板子连接与启动问候 |
| 板子接入 iPhone 后重连 | 已通过，重新上电后恢复 Wi-Fi 连接 |
| USB 输入就绪 | 零位移 `move 0 0` 命令返回成功，随后实际点击、滑动和输入均已执行 |
| AirPlay 服务发现 | 已通过，Bonjour 查询可发现 `iPhone Agent` |
| AirPlay 视频 | 已通过，持续取得 498×1080 JPEG 屏幕帧 |
| 自动校准 | 已通过，最终校准跨度为 w=504、h=1096 |
| 手动点击 | 已打开 General 页面，并验证小返回按钮及搜索框；手动点击已启用现有的视觉位置修正 |
| 手动滑动 | 已在 General 列表完成滑动，回传画面显示不同列表内容 |
| 键盘输入 | 已在设置搜索框输入 Bluetooth，实际出现对应搜索结果；通过清除按钮清理了测试文字 |
| DeepSeek 图片接口 | 已通过，deepseek-flash 将测试图片识别为 Red，耗时 1.06 秒 |
| DeepSeek 手机任务 | 已通过，从设置搜索页进入 General，18.17 秒结束，最终画面独立确认 About、Software Update、iPhone Storage 可见 |
| 软件检查 | 45 项测试通过、1 项可选 MCP 测试跳过；ruff 和脚本语法检查通过 |

## 投屏与点击问题的处理

手机握手标识为 `iPhone17,2`，报告 iPhone OS 27.0。首次只有 AirPlay 控制请求和定时反馈，没有屏幕镜像标记或视频流请求；重新从控制中心“两个重叠矩形”的屏幕镜像入口选择接收端后，屏幕帧正常到达。本次不需要切换 HEVC，也未证明存在 iOS 版本不兼容；音频输出列表中的设备勾选不能代替屏幕镜像验证。

GStreamer 全插件扫描和缺少 autodetect 插件问题已处理。运行日志位于 `.local/server.log`，此前诊断日志保留在同目录；该目录仅供本机用户访问，不作为公开文档或分发内容。

手动 API 原先只按相对鼠标映射点击，实测小按钮和底部搜索框出现偏差。现已让 `/api/tap` 使用项目已有的 `aim=True` 视觉定位修正，并复测小返回按钮、搜索框和清除按钮。对应并发控制测试也检查了精确定位参数，未改变原有操作互斥规则。

DeepSeek 短任务产生 1 次规划、3 次同位置点击和 1 次完成判断；最后一次点击明确指定了 precise。各模型请求约 0.90–1.23 秒，总任务约 18.17 秒。虽然最后达到目标，但重复点击仍值得优化，不能将命令返回 ok 或画面 changed 标记直接等同于命中目标。

当前中文输入法处于组合输入状态时，Cmd+A 没有全选，Backspace 只删除了末尾字符；随后使用界面清除按钮恢复空搜索框。已验证 ASCII 文字输入，不承诺所有键盘快捷键或中文输入都已可用。

## 验证证据

- 输入截图：`.local/validation/12-keyboard-input.jpg`。
- DeepSeek 最终画面：`.local/validation/15-deepseek-result.jpg`。
- 模型图片测试：`.local/validation/deepseek-image-check.json`。
- 手机任务结果：`.local/validation/deepseek-phone-task.json`。
- 完整任务轨迹：`.local/hidagent/traces/20261017-004816/`。

## 后续修复：Home、中文输入与 iPad

用户随后反馈反复无法返回主屏幕，并要求同时修复“美团”输入乱码，下一轮改用 iPad 验证。

- 原 Home 动作依赖相对鼠标从屏幕底边上滑，实测未稳定触发系统手势。改用独立的 `homescreen` 命令，不再依赖指针坐标。
- 第一版用了 `CONSUMER_CONTROL_HOME`（AC Home，0x0223），虽收到板子成功回执，设置画面仍未变化，判定失败。随后核对 [Apple Accessory Design Guidelines](https://developer.apple.com/accessories/Accessory-Design-Guidelines.pdf) 2026-09-21 版第 104 页、表 14-3：iPhone / iPad 的 Home Button 对应 Consumer 页 0x0C、Menu 0x0040。修正版改为该键值，按下 100 ms 后释放。
- 中文乱码来自把 UTF-8 字节直接交给 ASCII HID 键盘。固件新增整段字符校验，遇到非 ASCII 不输出任何字符；电脑端使用 pypinyin 把中文短词转成拼音按键。例如“美团”发送 `mei'tuan`，必须由系统中文拼音输入法产生候选词，再在画面中确认。Agent 的动作批次在中文输入后强制结束，不能在看候选词前继续执行 Enter、Space 或完成判断。
- 当前中文范围为每次 1–12 个纯汉字；中英混排分段输入，Emoji 和其他非 ASCII 文字仍需屏幕键盘。不会把拼音已发送等同于目标文字已输入。
- 固件增加 `key globe`（输入法选择）与 `key keyboard`（屏幕键盘切换）；对应 Apple Consumer 控件 0x029D 与 0x01AE。
- iPad 沿用相同固件和电脑架构；控制台投屏框改为按实际宽高比显示。校准保存画面比例，设备或横竖屏变化导致比例不匹配时拒绝使用旧校准。相同比例的新设备仍需手动重新校准。
- 合并固件编译通过：程序 964901 字节、静态全局内存 78696 字节；刷写后哈希校验通过。软件检查为 63 项通过、1 项可选 MCP 测试跳过，ruff 和 JavaScript 语法检查通过。

当前状态：已刷入合并固件并重启电脑端；新 Home 键、iPad 点按和中文候选词的真机验收尚待完成。旧版失败截图及本轮编译刷写日志保存在 `.local/home-fix/`，不可用先前 iPhone 短任务的成功替代本轮验收。

## 配置与下一步

Wi-Fi 密码只存在 Git 忽略的 `hardware/hid_bridge/secrets.h` 中。DeepSeek 配置位于 Git 忽略的 `server/.env`，使用 `deepseek-flash` 并关闭 thinking；API Key 由用户在本地填写，未写入固件、日志或本文。两份配置文件权限均为仅当前用户可读写。

控制台保持运行，手机停在 General 页面，Agent 已结束。网页的 Emergency stop 或 Esc 可停止任务。服务重启后通常需要重新选择屏幕镜像，校准数据保存在 `.local/hidagent/calibration.json`。

下一步应优先减少重复点击、检查输入法兼容性，再用多组短任务记录成功率。当前固件包含本机网络与服务器地址，只用于本轮验证；批量分享前仍需实现每位用户独立配网。手机 App 版和 USB-C 扩展硬件尚未验证。
