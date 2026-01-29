### Petri 网动画可视化工具

![image-20260122155140960](C:\Users\khand\AppData\Roaming\Typora\typora-user-images\image-20260122155140960.png)

基于 pygame 的交互式可视化工具，用于调试和演示连续 Petri 网环境。

**启动方式：**

```bash
python -m solutions.Continuous_model.test_env
```

**界面布局：**

- 水平排列的 4 个腔室矩形：`LP → PM1 → PM2 → LP_done`
- 3 个运输位矩形位于腔室之间：`d_PM1`、`d_PM2`、`d_LP_done`
- 箭头指示工艺流程方向

**晶圆状态颜色：**

| 位置类型           | 颜色 | 条件                                                         |
| ------------------ | ---- | ------------------------------------------------------------ |
| 加工腔室 (PM1/PM2) | 绿色 | 加工中 (`stay_time < processing_time`)                       |
| 加工腔室 (PM1/PM2) | 黄色 | 完成未报废 (`processing_time <= stay_time < processing_time + 20`) |
| 加工腔室 (PM1/PM2) | 红色 | 报废 (`stay_time >= processing_time + 20`)                   |
| 运输位             | 绿色 | 正常 (`stay_time < 10`)                                      |
| 运输位             | 红色 | 超时违规 (`stay_time >= 10`)                                 |

**交互控制：**

- **动作按钮**：点击执行对应变迁（灰色=不可用，蓝色=可用）
- **WAIT 按钮**：执行等待动作（始终可用）
- **Random 按钮**：随机选择一个可用动作
- **Reset 按钮**：重置环境
- **ESC 键**：退出程序

**信息面板：**

- 左下角：系统状态（当前时间、步数、累计奖励、上步奖励）
- 右下角：颜色图例说明





