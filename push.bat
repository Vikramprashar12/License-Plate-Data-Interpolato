@echo off
REM Batch script to automate git push to Vikram branch

REM Navigate to your repository's directory
cd /d "C:\Users\Vikram Prashar\OneDrive - Toronto Metropolitan University\Documents\Coding Adventures\CPS_843_Project"

REM Check out the Vikram branch
git checkout Vikram

REM Stage all changes
git add .

REM Commit changes with a message
set /p commitMsg="Enter commit message:"
git commit -m "%commitMsg%"

REM Push changes to the Vikram branch
git push origin Vikram

REM Confirm completion
echo Changes have been pushed to the Vikram branch.
pause
