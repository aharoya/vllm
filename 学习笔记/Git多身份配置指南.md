# Git 多身份配置指南

工作时如何处理公司项目和个人 GitHub 项目使用不同 git 身份的问题。

## 原理

git 配置分三级，优先级从高到低：

```
仓库级 (.git/config)  >  全局级 (~/.gitconfig)  >  系统级 (/etc/gitconfig)
```

低优先级的配置不会消失，只是被更高优先级覆盖。所以：

- 全局用公司邮箱 → 公司项目自动用公司身份
- 个人仓库本地覆盖 → 仅该项目用个人身份

## 快速配置（新项目接入）

```bash
cd 你的项目目录

# 设置仅当前仓库生效的用户名和邮箱
git config user.name "你的GitHub用户名"
git config user.email "你的GitHub邮箱"

# 验证
git config user.name   # 应显示 GitHub 用户名
git config user.email  # 应显示 GitHub 邮箱
```

> 注意：**不要加 `--global`**，加了就成全局设置了。

## 验证配置不冲突

```bash
# 查看当前仓库的配置
git config user.name
git config user.email

# 查看全局配置（不影响，仍然保留）
git config --global user.name
git config --global user.email
```

## Q&A

**Q: commit 后 GitHub 上只显示名字，没有头像跳转？**

A: commit 的邮箱与 GitHub 账号绑定邮箱不匹配。两种解决方式：

1. 把公司邮箱添加到 GitHub 账号（Settings → Emails → Add email address）
2. 在对应仓库改用 GitHub 注册邮箱

**Q: 不小心用了全局邮箱提交了怎么办？**

A: 可以用 `git commit --amend --author="name <email>"` 修改最近一次提交的作者信息。

**Q: HTTPS push 每次都要输密码？**

A: 在仓库远程地址中嵌入 token（仅当前仓库生效）：
```bash
git remote set-url origin https://<token>@github.com/用户名/仓库名.git
```

**Q: GitHub 直连超时需要代理？**

A: 在 URL 中插入代理前缀：
```bash
git remote set-url origin https://<token>@代理地址/https://github.com/用户名/仓库名.git
```
