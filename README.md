# OpenClaw Runpod Minimal Train

## 1. Colab 验证

```bash
cd project
bash scripts/run_local.sh
```

确认输出了 `outputs/min_run/checkpoint.pt`。

## 2. Runpod terminal 验证

```bash
cd project
bash scripts/run_runpod.sh
```

确认输出了 `outputs/min_run/checkpoint.pt`。

## 3. OpenClaw / SSH 验证

先测试 SSH：

```bash
cd project
bash scripts/test_ssh.sh user@host
```

然后远程执行同一训练脚本：

```bash
ssh user@host 'cd /path/to/project && bash scripts/run_runpod.sh'
```
