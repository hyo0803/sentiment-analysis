# Removing a file added in the most recent unpushed commit

If the file was added with your most recent commit, and you have not pushed to GitHub.com, you can delete the file and amend the commit:

1. Open Terminal.
2. Change the current working directory to your local repository.
3. To remove the file, enter __git rm --cached__:
```
$ git rm --cached GIANT_FILE
# Stage our giant file for removal, but leave it on disk
```

4. Commit this change using __--amend -CHEAD__:
```
$ git commit --amend -CHEAD
# Amend the previous commit with your change
# Simply making a new commit won't work, as you need
# to remove the file from the unpushed history as well
```

5. Push your commits to GitHub.com:
```
$ git push
# Push our rewritten, smaller commit
```
or 
```
git push --set-upstream origin [branch_name]
```

# Как правильно хранить большие файлы в ML-проектах
Использовать Git LFS (Large File Storage)
1. Установка:
```
brew install git-lfs
git lfs install
> Git LFS initialized.
```

2. Отслеживание файлов:
```
git lfs track "data/*.csv"
```
> Это обновит (или создаст) файл .gitattributes. Не забудьте его закоммитить позже!

3. Переписать историю, чтобы перевести файлы в LFS
```
# Перевести файлы по расширению во всей истории
git lfs migrate import --include="*.csv,*.joblib" --everything
```
Или, если файлы в определённой папке:
```
git lfs migrate import --include="data/,models/" --everything
```
Флаг --everything означает: обработать все ветки и теги. Если нужна только текущая ветка — убрать флаг. 

Эта команда (__git lfs migrate import__):
- Перепишет коммиты, в которых есть указанные файлы
- Заменит их на указатели LFS
- Автоматически обновит .gitattributes