# Custom message for release
RELEASE_MESSAGE="Latest release"

# Finishing the Git Flow release
git flow release finish -m "$RELEASE_MESSAGE"

# I suppose we can make it fully automatic.
git push origin develop
git push origin master
git push --tags
