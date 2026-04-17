# 0001 - 仓库重整与数据外置

状态：Accepted  
日期：2026-04-17

## 背景（Context）

项目从 Windows 迁移到 Linux 后，原仓库同时承载代码、数据、输出和虚拟环境，导致：

- 路径耦合严重（大量 Windows 绝对路径）
- 仓库体积过大，不利于同步与协作
- 双系统共享数据不便
- 云端训练迁移成本高

## 决策（Decision）

- 仓库聚焦“代码与文档”，大体量资源外置到共享盘。
- 保留稳定入口名 `res` / `output` / `adapter_strategy_v1/checkpoints`，通过软连接映射到：
  - `/mnt/data/aigc_data/datasets/res`
  - `/mnt/data/aigc_data/outputs/output`
  - `/mnt/data/aigc_data/checkpoints/adapter_strategy_v1/checkpoints`
- 环境管理从 `venv` 切换到 Miniconda，环境定义固化到 `environment.yml`。
- 历史代码不删除，迁移到 `archive_legacy/`。

## 备选方案（Alternatives）

1. 继续将数据放在仓库内。  
   - 问题：体积大、迁移慢、双系统协作成本高。
2. 直接把所有脚本改为硬编码绝对路径（`/mnt/data/...`）。  
   - 问题：可移植性差，云端训练要二次改代码。

## 影响与后果（Consequences）

正向：

- 代码路径保持稳定，脚本改动最小。
- 数据与输出可跨 Linux/Windows 共享。
- 未来迁移云端可通过重新映射软连接快速适配。

代价：

- 需要确保外置盘挂载正常。
- 新环境初始化时需要先执行目录映射脚本。

## 备注

- 对应执行脚本：`scripts/reorganize_linux.sh`
