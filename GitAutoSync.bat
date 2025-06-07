@echo off
setlocal enabledelayedexpansion
set LOCAL_DIR=F:\stroke_prognosis
set REMOTE_REPO=git@github.com:xiaobo1789/stroke_prognosis.git
set BRANCH=main
cd /d "%LOCAL_DIR%"
echo 正在拉取远程更新...
git pull "%REMOTE_REPO%" "%BRANCH%"
git diff --quiet HEAD
if %errorlevel% equ 0 (
    echo 无修改，跳过同步
) else (
    echo 检测到修改，开始提交...
    git add .
    set COMMIT_MSG=Auto sync: !date! !time!
    git commit -m "!COMMIT_MSG!"
    echo 正在推送...
    git push "%REMOTE_REPO%" "%BRANCH%"
    echo 同步完成：!COMMIT_MSG!
)
echo [%date% %time%] 同步结果：%errorlevel% >> "%LOCAL_DIR%\GitSyncLog.txt" 2>&1
endlocal