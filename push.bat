@echo off
REM Batch script to automate git push to Hayaan's branch

REM Navigate to your repository's directory
cd /d "C:\Users\t2xpl\Documents\Projects\CPS_843_Project"

REM Check out the Hayaan branch
git checkout Hayaan

REM Stage all changes
git add .

REM Commit changes with a message
set /p commitMsg="Enter commit message:"
git commit -m "%commitMsg%"

REM Push changes to the Hayaan branch
git push origin Hayaan

REM Confirm completion
echo Changes have been pushed to the Hayaan branch.
pause