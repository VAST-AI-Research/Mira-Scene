# Mira-Scene 结果可视化

[English](README.md) | [简体中文](README_zh-CN.md)

该目录提供只读的推理结果 Web 查看器。

## 启动

输入可以是包含多个 case 的结果根目录，也可以是单个 case：

```bash
python visualization/visualize_results.py /path/to/results --port 8000
```

服务默认监听 `0.0.0.0`。启动后访问：

```text
http://127.0.0.1:8000/
http://SERVER_IP:8000/
```

若只允许本机访问，可增加 `--host 127.0.0.1`。服务器、防火墙或 Kubernetes
仍需开放对应端口；该服务没有身份认证，请勿暴露到不可信网络。

## 展示内容

下拉菜单用于切换 case。每个 case 展示一行：

1. 输入图：`input/scene.png`；
2. 分割图：`CCM/rgb_mask.png` 的右半部分；
3. 交互式 3D 场景。

3D 场景支持不同 mesh/depth 结果切换，优先读取
`scene_with_floor.glb`，否则读取 `scene.glb`。如果存在
`environment/environment_equirect.*`，会作为背景和环境光，并可通过滑条
调整水平方向。

图片会完整缩放到与 3D 窗口相同的高度，不进行裁剪。3D 内容点击后才加载，
最多同时保留六个 viewer；超出时最早打开的会卸回 Load 3D 按钮，可再次加载。

点击已加载的 3D 窗口即可激活键盘控制：`W/S` 前后移动，`A/D` 左右移动，
`Q/E` 下上移动，按住 `Shift` 加速。只有当前聚焦的 3D 窗口会响应按键。

## 支持的目录结构

```text
results/<case>/
├── input/scene.png
├── CCM/rgb_mask.png
├── scene/<backend>/<method>_depth/
│   ├── scene_with_floor.glb
│   └── scene.glb
└── environment/environment_equirect.png
```

