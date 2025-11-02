# ICE2607teamProject
一些链接 <a href = "https://github.com/detectRecog/CCPD?tab=readme-ov-file"> 数据库 </a>
有能力可以下载一下




## AI写的废话

### 最核心的规则



**任何人都不要直接往 `main` 分支上提交（push）代码。**

`main` 分支只用来存放所有人合并好的、能正常运行的“最终版本”。

------



### 1. 组长（或指定一个同学）的初始设置



开始写代码前，需要先设置好仓库：

1. **建仓库：** 组长在 GitHub 上建一个**私有**仓库（Private Repository）。
2. **加组员：** 在仓库的 `Settings` > `Collaborators` 里，把所有组员的 GitHub 账号都加进来。
3. **保护 `main` 分支（最重要的一步）：**
   - 进入 `Settings` > `Branches`。
   - 点击 `Add branch protection rule`。
   - 在 "Branch name pattern" 里填入 `main`。
   - 勾选 **"Require a pull request before merging"**（必须通过 PR 才能合并）。
   - 勾选 **"Require approvals"**（需要有人批准），并把数量设为 `1`。
   - 点击 `Create` 保存。
   - *这样做以后，就没人能不小心把 `main` 分支搞坏了。*
4. **添加 `.gitignore` 文件：**
   - 在项目根目录创建一个叫 `.gitignore` 的文件。
   - 这能防止你们把本地的 IDE 配置（如 `.vscode/`, `.idea/`）、Python 缓存 (`__pycache__/`)、数据集和训练好的模型 (`.pt`, `.weights`, `dataset/`) 等不需要共享的文件传上去。

------



### 2. 组员的协作流程（重点）



这里分为两部分：**A. 第一次拉取项目** 和 **B. 之后的日常开发**。



#### A. 第一次：克隆项目到本地 (Clone) - (只需做一次！)



这是你把 GitHub 上的项目复制到你电脑上的第一步，每个组员在项目开始时只需要做一次。

1. 让你的组长把仓库的 URL 发给你。或者你自己去 GitHub 上的仓库主页。

2. 点击那个绿色的 `<> Code` 按钮。

3. 复制 `HTTPS` 或 `SSH` 链接（推荐用 HTTPS，比较简单）。

4. 打开你电脑的终端（Terminal 或 Git Bash），`cd`到你平时存放代码的文件夹。

5. 运行 `git clone` 命令：

   Bash

   ```
   # 把 [你复制的链接] 换成你上一步复制的真实链接
   git clone [你复制的链接]
   ```

6. 好了！现在这个项目已经在你的电脑上了。



#### B. 日常工作流：开发新功能（每次都要做）



当你准备开始做一个新功能时（比如“开发GUI界面”），请严格按以下步骤操作：

第1步：同步最新的代码 (Sync)

在你开始写新功能前，永远先保证你本地的 main 分支是最新版。

Bash

```
# 1. 切换回 main 分支
git checkout main
# 2. 从 GitHub 上拉取最新的代码
git pull origin main
```

第2步：开一个新的分支（Branch）

你所有的开发工作都在自己的分支上进行，不要动 main。

Bash

```
# 3. 从 main 分支上创建一个新分支
#    (分支名要清晰，比如 feature/gui-interface 或 bugfix/fix-login)
git checkout -b feature/gui-interface 
```

现在，你就在 `feature/gui-interface` 这个“沙盒”里了，可以随便写代码，不会影响到其他人。

第3步：写代码和提交（Commit & Push）

在你的新分支上写代码。当你完成了一个小功能点（比如“完成了主窗口布局”），就 commit 一次。

Bash

```
# 4. 添加你修改过的文件
git add .
# 5. 提交你的修改，-m 后面写清楚你做了什么
git commit -m "feat: 完成GUI主窗口布局"
# 6. 把你的分支推送到 GitHub 仓库
git push origin feature/gui-interface
```

第4步：创建拉取请求（Pull Request - PR）

当你的功能开发完成后，就可以请求把它合并到 main 分支了。

1. 推送（Push）成功后，你打开 GitHub 仓库主页，会看到一个黄色的提示条，点击 `Compare & pull request` 按钮。
2. **填写 PR：**
   - **标题：** 清晰地写明这个 PR 是做什么的（比如：`[功能] 完成GUI界面`）。
   - **描述：** 简单说明你做了哪些工作。
   - **Reviewers：** 在右侧栏，@ 你的**至少一个组员**来检查你的代码。
3. 点击 `Create pull request`。

------



### 3. 代码审查 (Review) 和合并



这个流程是保证代码质量的关键：

1. **审查 (Review)：** 你的组员会收到通知。他们需要打开这个 PR，点击 `Files changed` 选项卡，检查你写的代码。
2. **反馈：**
   - **如果没问题：** 组员点击 `Review changes` > `Approve`（批准）。
   - **如果有问题：** 组员会在代码的特定行留言，或者点击 `Request changes`。
3. **修改：** 如果被要求修改，你就在**本地的同一个分支** (`feature/gui-interface`) 上继续修改代码，然后再次 `commit` 和 `push`。PR 会自动更新你的修改。
4. **合并 (Merge)：**
   - 当你的 PR 获得了组员的 `Approve`（并且没有冲突），那个绿色的 `Merge pull request` 按钮就可以点了。
   - 点击它，你的代码就安全地合并到 `main` 分支里了。
   - 之后可以删除这个功能分支。

------



### 4. 万一遇到“合并冲突”怎么办？



**什么是冲突？** 就是你和你的组员，在差不多的时间里，修改了**同一个文件的同一行代码**。Git 不知道该听谁的。

**怎么解决（推荐在本地）：**

1. 假设你在 `feature/gui-interface` 分支上，准备推代码，或者准备合并 PR，但 Git 提示你有冲突。

2. **第一步：** 先把你本地的 `main` 更新到最新：

   Bash

   ```
   git checkout main
   git pull origin main
   ```

3. **第二步：** 切回你自己的功能分支：

   Bash

   ```
   git checkout feature/gui-interface
   ```

4. **第三步：** 把最新的 `main` 合并到你**当前**的分支：

   Bash

   ```
   git merge main
   ```

5. **第四步：** 终端会提示你 `CONFLICT!` 并告诉你哪个文件冲突了。

6. **第五步：** 打开那个冲突的文件（用 VS Code 等编辑器）。你会看到类似这样的标记：

   ```
   <<<<<<< HEAD
   (这里是你的代码)
   =======
   (这里是 main 分支上新拉下来的代码)
   >>>>>>> main
   ```

7. **第六步：** **手动编辑它！**

   - 和你的组员商量一下，决定保留哪部分代码，或者怎么把两部分代码结合起来。
   - **删除掉所有 `<<<<<<<`、`=======`、`>>>>>>>` 这些标记**，只留下你们商量好的最终代码。

8. **第七步：** 保存文件，然后 `commit` 这个“解决冲突”的修改：

   Bash

   ```
   git add .
   git commit -m "fix: 解决了合并冲突"
   ```

9. **第八步：** 再次 `push` 你的分支，现在冲突就解决了。

   Bash

   ```
   git push origin feature/gui-interface
   ```

------



