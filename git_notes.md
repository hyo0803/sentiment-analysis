# Removing a file added in the most recent unpushed commit

If the file was added with your most recent commit, and you have not pushed to GitHub.com, you can delete the file and amend the commit:

1. Open Terminal.
2. Change the current working directory to your local repository.
3. To remove the file, enter __git rm --cached__:
> $ git rm --cached GIANT_FILE
> # Stage our giant file for removal, but leave it on disk

4. Commit this change using __--amend -CHEAD__:
> $ git commit --amend -CHEAD
> # Amend the previous commit with your change
> # Simply making a new commit won't work, as you need
> # to remove the file from the unpushed history as well

5. Push your commits to GitHub.com:
> $ git push
> # Push our rewritten, smaller commit
or 
> git push --set-upstream origin [branch_name]
