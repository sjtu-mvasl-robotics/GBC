for branch in $(git branch -r | grep -v '\->'); do
    git branch --track ${branch#origin/} $branch || true
done